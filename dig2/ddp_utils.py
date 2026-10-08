"""Multi-GPU (DDP) topology resolution and collective helpers.

Every function here is designed around one invariant, which is the whole contract of the
DDP work: **at world size 1 nothing in this module changes behaviour.** The reducers
return their input object untouched, the gather helpers return the input list, and
`is_ddp()` is `False` so callers skip their DDP branches entirely. That is what keeps a
`general.gpus=[N]` run byte-for-byte identical to the pre-DDP code -- no new collective,
no new tensor allocation, no new RNG draw, no new float op.

House prior art for this "opt-in path returns before the old code is touched" pattern:
`NodeBudgetWeightedBatchSampler.__iter__`'s `size_unbiased` early return
(`datasets/abstract_dataset.py`), and `sample.rng_safe_per_domain`.

WHY TOPOLOGY IS READ FROM THE ENVIRONMENT, NOT FROM torch.distributed
---------------------------------------------------------------------
`AbstractDataModule` is constructed inside `main()` *before* the `Trainer` exists, and the
node-budget batch samplers must know `(rank, world_size)` at construction time so they can
install the rank-strided view. At that point `torch.distributed` is not initialized, so
`dist.get_rank()` is unavailable.

Lightning's DDP subprocess launcher (`_SubprocessScriptLauncher`) sets `MASTER_ADDR`,
`MASTER_PORT`, `NODE_RANK`, `LOCAL_RANK` and `WORLD_SIZE` in each child's environment
*before* the training script is re-executed, so the env vars are already correct by the
time `main()` runs in a child. The parent process is rank 0 and has no `LOCAL_RANK` set,
which correctly resolves to 0. The config fallback (`len(cfg.general.gpus)`) covers the
parent process, where `WORLD_SIZE` is not yet set but the intended world size is known
statically.
"""

import os

import torch

try:  # OmegaConf is always present in this repo, but keep the import defensive.
    from omegaconf import ListConfig
except Exception:  # pragma: no cover
    ListConfig = ()


def configured_world_size(cfg) -> int:
    """World size implied by `cfg.general.gpus`, with no environment involved.

    `general.gpus` is polymorphic by long-standing convention:
      * a list of DEVICE IDS -- `gpus: [7]` pins card 7, `gpus: [2,5,7]` is 3-way DDP
      * a plain int -- a device COUNT (the `general_default.yaml` form)
    """
    gpus = getattr(getattr(cfg, "general", None), "gpus", None)
    if isinstance(gpus, (list, tuple, ListConfig)):
        return max(1, len(gpus))
    try:
        return max(1, int(gpus))
    except (TypeError, ValueError):
        return 1


def ddp_topology(cfg=None):
    """Return `(rank, world_size)` without requiring `torch.distributed` to be initialized.

    Resolution order, first hit wins:
      1. An initialized process group  -> the authoritative answer.
      2. `RANK` + `WORLD_SIZE` env     -> torchrun, or a Lightning child after setup.
      3. `LOCAL_RANK` env + `cfg`      -> a Lightning child on a single node.
      4. `cfg` alone                   -> the parent process, which is rank 0.
      5. `(0, 1)`                      -> no information at all; single process.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()

    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world is not None:
        try:
            return int(env_rank), max(1, int(env_world))
        except ValueError:
            pass

    rank = 0
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            rank = int(local_rank)
        except ValueError:
            rank = 0

    world_size = configured_world_size(cfg) if cfg is not None else 1
    if env_world is not None:
        try:
            world_size = max(world_size, int(env_world))
        except ValueError:
            pass
    return rank, max(1, world_size)


def ddp_world_size(cfg=None) -> int:
    return ddp_topology(cfg)[1]


def is_ddp(cfg=None) -> bool:
    """True only when more than one rank participates. The single gate for every branch."""
    return ddp_world_size(cfg) > 1


def _collectives_ready() -> bool:
    """A live process group. Collectives are only legal once Lightning has set one up."""
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def all_reduce_sum_(tensor):
    """In-place SUM all-reduce. Returns `tensor` UNCHANGED when there is no process group.

    The no-op return is load-bearing: callers multiply by `local / global`, which is
    exactly `1.0` at world size 1, so the single-GPU arithmetic is unchanged.
    """
    if _collectives_ready():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def all_reduce_min_(tensor):
    """In-place MIN all-reduce, for turning a rank-local predicate into a global one.

    Used to make "skip this batch" a unanimous decision: a bare per-rank `return` from
    `training_step` skips `backward()` on that rank alone and deadlocks every other rank
    in the gradient all-reduce.
    """
    if _collectives_ready():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return tensor


def barrier():
    """Block until every rank arrives. No-op without a process group.

    Needed wherever one rank does work the others skip (e.g. rank-0-only validation
    sampling): without it the idle ranks race ahead into the next training epoch and the
    ranks drift apart in the dataloader stream.
    """
    if _collectives_ready():
        torch.distributed.barrier()


def _walk_metric_containers(obj, seen=None, depth=0):
    """Yield every torchmetrics Metric reachable from `obj`, including through plain
    dict/list/tuple attributes.

    `nn.Module.modules()` is NOT sufficient here. `MixedMetricDispatcher` -- the evaluator
    this project's mixed molecular+structural arms actually build -- keeps its per-domain
    evaluators in a PLAIN DICT:

        self.molecular_evaluator_by_domain = molecular_evaluator_by_domain or {}

    `nn.Module.__setattr__` only registers values that are themselves Modules, so a dict of
    Modules is invisible to `.modules()`. Walking only `.modules()` therefore reports zero
    metrics on an arm that in fact holds four per molecular domain (qm9 and zinc =>
    eight), and the rank-0-only scoring path then hangs on the first `.compute()`.
    """
    if seen is None:
        seen = set()
    if depth > 6 or id(obj) in seen:
        return
    seen.add(id(obj))
    try:
        from torchmetrics import Metric
    except Exception:  # pragma: no cover
        return
    if isinstance(obj, Metric):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_metric_containers(v, seen, depth + 1)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _walk_metric_containers(v, seen, depth + 1)
        return
    if isinstance(obj, torch.nn.Module):
        for child in obj.children():
            yield from _walk_metric_containers(child, seen, depth + 1)
        # plus any dict/list attribute holding Modules or Metrics
        for v in vars(obj).values():
            if isinstance(v, (dict, list, tuple, set)):
                yield from _walk_metric_containers(v, seen, depth + 1)
        return


def disable_metric_sync(module):
    """Turn OFF cross-rank sync on every torchmetrics Metric reachable from `module`.

    Use when exactly ONE rank computes a metric over data it already holds in full. The
    default `sync_on_compute=True` makes `.compute()` a collective, so a single-rank call
    would hang the others; and were all ranks to call it, the states would be summed and
    the result double-counted.

    Traversal goes through plain containers as well as registered submodules -- see
    `_walk_metric_containers` for why `.modules()` alone silently finds nothing on the
    mixed-domain arms.

    Returns the number of metrics changed. No-op without a process group.
    """
    if not _collectives_ready():
        return 0
    n = 0
    for m in _walk_metric_containers(module):
        m.sync_on_compute = False
        n += 1
    return n


def all_gather_objects(obj):
    """Gather one picklable object per rank into a rank-ordered list.

    Returns `[obj]` when there is no process group, so a caller that flattens the result
    sees exactly its own input at world size 1.
    """
    if not _collectives_ready():
        return [obj]
    buf = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(buf, obj)
    return buf


def shard_count(total: int, rank: int, world_size: int) -> int:
    """Split `total` across ranks so the shards sum to EXACTLY `total`.

    Earlier ranks absorb the remainder. Used for per-domain generation budgets, where the
    global total must be preserved because downstream metric denominators assume it.
    """
    total = int(total)
    if world_size <= 1:
        return total
    return total // world_size + (1 if rank < (total % world_size) else 0)


def ddp_preflight(cfg):
    """Raise if `cfg` names a feature that is not DDP-safe. No-op at world size 1.

    Everything listed here fails SILENTLY under DDP rather than crashing, so the run would
    finish and produce plausible-looking numbers built on a broken invariant.
    """
    rank, ws = ddp_topology(cfg)
    if ws == 1:
        return

    g = getattr(cfg, "general", None)
    t = getattr(cfg, "train", None)
    problems = []

    def _on(obj, *path, default=False):
        cur = obj
        for k in path:
            cur = getattr(cur, k, None)
            if cur is None:
                return default
        return bool(cur)

    if _on(g, "theta_head", "enabled"):
        problems.append(
            "general.theta_head.enabled=true is NOT DDP-safe. ThetaHead is built LAZILY "
            "inside training_step (_get_theta_head), i.e. AFTER DDP constructed its "
            "reducer, so its parameters are never all-reduced and every rank silently "
            "trains a private copy that drifts from the others. The code already "
            "hand-patches the OPTIMIZER for this; nothing patches the reducer, and torch "
            "does not support doing so. Run theta-head experiments on one GPU."
        )
    if _on(g, "theta_bandit", "enabled"):
        problems.append(
            "general.theta_bandit.enabled=true is not DDP-safe: theta_bandit_state.json is "
            "an unguarded read-modify-write from every rank (last writer wins, posterior "
            "counts lost), and arm selection happens per-rank so ranks would sample under "
            "different theta."
        )
    if str(getattr(getattr(cfg, "model", None), "type", "")).strip().lower() == "sparse_full":
        problems.append(
            "model.type=sparse_full is not covered by the DDP work. It has a SECOND, "
            "duplicated training step (_training_step_sparse_full) with its own loss "
            "structure and its own copy of the prototype EMA; neither received the "
            "count-weighted loss scaling nor the prototype all-reduce, both of which would "
            "ship untested. Use model.type=gt under DDP."
        )
    mode = str(getattr(t, "batching_mode", "")).strip().lower()
    if mode in {"domain_round_robin", "domain_balanced_cycling"}:
        problems.append(
            f"train.batching_mode={mode} is declared in configs but UNIMPLEMENTED -- "
            "abstract_dataset.py silently falls through to the plain shuffle loader, which "
            "has no rank-aware sharding, so every rank would train on IDENTICAL batches "
            "(N x the compute for a 1 x effective batch). Use node_budget."
        )
    try:
        if float(getattr(t, "ema_decay", 0) or 0) > 0:
            problems.append(
                "train.ema_decay>0 references utils.EMA, which does not exist in "
                "src/utils.py -- it raises AttributeError at callback construction."
            )
    except (TypeError, ValueError):
        pass
    if _on(g, "gpu_guard_enabled"):
        problems.append(
            "general.gpu_guard_enabled=true reserves VRAM from a device index resolved "
            "before Lightning assigns per-rank devices; not validated under DDP."
        )

    if mode in {"node_budget", "domain_node_budget"}:
        try:
            mg = int(getattr(t, "min_graphs_per_batch", 1))
        except (TypeError, ValueError):
            mg = 1
        if mg < ws:
            problems.append(
                f"train.min_graphs_per_batch={mg} < world_size={ws}. Stride sharding gives "
                f"rank r the graphs batch[r::{ws}], so a global batch with fewer than {ws} "
                f"graphs leaves a rank EMPTY and deadlocks the gradient all-reduce. Set it "
                f">= {ws} and raise train.max_nodes_per_batch so the node budget, not the "
                f"graph floor, stays the binding constraint."
            )

    if problems:
        raise RuntimeError(
            f"[ddp_preflight] {len(problems)} config setting(s) are not safe at "
            f"world_size={ws}:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    print(f"[ddp] preflight OK at world_size={ws}", flush=True)
