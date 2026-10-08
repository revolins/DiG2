import copy
import os
import random
from collections.abc import Mapping

import torch
from torch.utils.data import SequentialSampler
from torch_geometric.loader import DataLoader

from dig2.ddp_utils import ddp_topology
from torch_geometric.data.lightning import LightningDataset
from tqdm import tqdm

from dig2 import utils
from dig2.data.dataset_utils import DistributionNodes, load_pickle, save_pickle


def _normalize_domain_name(name):
    return str(name).strip().lower()


def _extract_domain_meta_from_dataset(dataset):
    """Schema-first domain metadata extraction with compatibility fallbacks."""
    raw_names = list(getattr(dataset, "dataset_names", []) or [])
    domain_names = [_normalize_domain_name(n) for n in raw_names if str(n).strip()]
    domain_indices = {name: [] for name in domain_names}

    if hasattr(dataset, "domain_indices"):
        for name, indices in dict(getattr(dataset, "domain_indices", {})).items():
            norm_name = _normalize_domain_name(name)
            domain_indices.setdefault(norm_name, []).extend(list(indices))
        if len(domain_names) == 0:
            domain_names = list(domain_indices.keys())
        domain_counts = {d: len(domain_indices.get(d, [])) for d in domain_names}
        return {
            "domain_names": domain_names,
            "domain_indices": domain_indices,
            "domain_counts": domain_counts,
        }

    for idx in range(len(dataset)):
        sample = dataset[idx]
        name = getattr(sample, "dataset_name", None)
        if name is None:
            continue
        norm_name = _normalize_domain_name(name)
        if norm_name not in domain_indices:
            domain_indices[norm_name] = []
            domain_names.append(norm_name)
        domain_indices[norm_name].append(idx)

    domain_counts = {d: len(domain_indices.get(d, [])) for d in domain_names}
    return {
        "domain_names": domain_names,
        "domain_indices": domain_indices,
        "domain_counts": domain_counts,
    }


def _extract_node_counts_from_dataset(dataset):
    """Return list[int] of per-sample node counts from pre-computed metadata, or None."""
    n_nodes_attr = getattr(dataset, "n_nodes_by_global_idx", None)
    if n_nodes_attr is None:
        return None
    if isinstance(n_nodes_attr, list):
        return n_nodes_attr
    return [int(n) for n in n_nodes_attr]


class NodeBudgetWeightedBatchSampler(torch.utils.data.Sampler):
    """Variable-size batch sampler with domain-balanced weights and a node budget.

    Equivalent to per-sample weights = 1/|domain|: each domain gets equal total
    draw probability regardless of dataset size. Accumulates samples until total
    node count reaches max_nodes_per_batch, then yields the batch.  Yields
    exactly steps_per_epoch batches per epoch, with a fresh random seed each
    epoch to avoid repeating the same sequence.

    This achieves ~equal expected node-count exposure per domain per batch, which
    is the natural normalization for node/edge-wise cross-entropy losses.
    """

    def __init__(
        self,
        domain_indices,
        n_nodes,
        max_nodes_per_batch,
        min_graphs_per_batch=1,
        steps_per_epoch=250,
        seed=0,
        size_unbiased=False,
        domain_draw_weights=None,
    ):
        super().__init__(None)
        self.n_nodes = n_nodes
        self.max_nodes = int(max_nodes_per_batch)
        self.min_graphs = int(min_graphs_per_batch)
        self.steps = int(steps_per_epoch)
        self.seed = int(seed)
        self.size_unbiased = bool(size_unbiased)
        # Per-domain draw weights. None => uniform over domain NAMES, i.e. the historical
        # behaviour. Only consulted on the size-unbiased path, so the default __iter__ is
        # left literally untouched.
        self._domain_draw_weights_cfg = dict(domain_draw_weights or {})
        self._call_count = 0

        self._active_domain_names = sorted(
            [d for d, idxs in domain_indices.items() if len(idxs) > 0]
        )
        self._domain_idx_lists = {
            d: list(domain_indices[d]) for d in self._active_domain_names
        }

        if self._domain_draw_weights_cfg:
            self._domain_weights = [
                float(self._domain_draw_weights_cfg.get(d, 1.0))
                for d in self._active_domain_names
            ]
            if min(self._domain_weights) < 0 or sum(self._domain_weights) <= 0:
                raise ValueError(
                    f"domain_draw_weights must be non-negative with positive sum, got "
                    f"{dict(zip(self._active_domain_names, self._domain_weights))}"
                )
            _tot = sum(self._domain_weights)
            print(
                "[node_budget] domain_draw_weights active: "
                + ", ".join(
                    f"{d}:{w/_tot:.3f}"
                    for d, w in zip(self._active_domain_names, self._domain_weights)
                )
            )
        else:
            self._domain_weights = None

        n_domains = len(self._active_domain_names)
        if n_domains > 0:
            max_graph_size = max(n_nodes) if n_nodes else 0
            if max_graph_size > self.max_nodes:
                print(
                    f"[node_budget] Warning: max_nodes_per_batch={self.max_nodes} is smaller than "
                    f"the largest graph ({max_graph_size} nodes). "
                    f"Oversized graphs will be skipped — increase max_nodes_per_batch to include them."
                )
            expected_nodes_each = self.max_nodes / n_domains
            parts = []
            for d in self._active_domain_names:
                idxs = self._domain_idx_lists[d]
                mean_n = sum(n_nodes[i] for i in idxs) / len(idxs)
                parts.append(f"{d}:{expected_nodes_each / mean_n:.1f}")
            print(
                f"[node_budget] max_nodes_per_batch={self.max_nodes}, "
                f"steps_per_epoch={self.steps} | "
                f"expected graphs/batch (~{expected_nodes_each:.0f} nodes/domain): "
                + ", ".join(parts)
            )

    def __len__(self):
        return self.steps

    def _iter_size_unbiased(self):
        """Size-unbiased variant of __iter__ (opt-in: train.node_budget_size_unbiased).

        The default loop DISCARDS a drawn graph that does not fit the remaining budget
        ("yield current batch without this graph"). Acceptance therefore falls
        monotonically with graph size, and because draws are i.i.d. with replacement and
        there is no epoch pass, the discarded mass is never repaid -- the consumed
        node-count distribution converges to a rejection-distorted one, permanently.
        Measured on the unified cache at max_nodes_per_batch=256: sbm's consumed mean size sits
        7.5 nodes below its true mean and the n>=140 band is consumed at 0.730x its true
        share, while fixed-size domains (planar/tree at 64 nodes) are unaffected.

        This variant CARRIES a non-fitting draw over to the next batch instead of
        discarding it, so every drawn graph is consumed exactly once and the consumed
        distribution equals the true one by construction -- no rejection sampling, no
        importance weights. Simulated on the same cache: every band ratio lands in
        [0.992, 1.007] and mean nodes/batch is preserved (216.8 -> 217.1), which is the
        quantity that matters because the losses are node/edge-wise. Graphs/batch falls
        ~11% (4.61 -> 4.10) -- the intended consequence of carrying the larger graphs the
        default path drops.

        A graph larger than max_nodes cannot share a batch, so it is emitted as a solo
        over-budget batch rather than skipped forever (the default path's `continue` makes
        such graphs structurally unreachable). Watch peak memory if the data contains
        graphs above the budget; in the unified cache sbm maxes at 187 against a 256 budget, so
        that path does not fire there.
        """
        rng = random.Random(self.seed + self._call_count)
        self._call_count += 1
        domain_names = self._active_domain_names
        domain_idx_lists = self._domain_idx_lists
        min_graphs = max(1, int(getattr(self, "min_graphs", 1)))
        carry = None
        for _ in range(self.steps):
            batch, total_nodes = [], 0
            if carry is not None:
                batch.append(carry)
                total_nodes += self.n_nodes[carry]
                carry = None
            while True:
                if total_nodes >= self.max_nodes and len(batch) >= min_graphs:
                    break
                if self._domain_weights is None:
                    domain = rng.choice(domain_names)
                else:
                    domain = rng.choices(domain_names, weights=self._domain_weights, k=1)[0]
                idx = rng.choice(domain_idx_lists[domain])
                n = self.n_nodes[idx]
                if total_nodes + n > self.max_nodes and len(batch) >= min_graphs:
                    carry = idx          # deferred to the next batch, NOT discarded
                    break
                batch.append(idx)
                total_nodes += n
            yield batch

    def __iter__(self):
        if getattr(self, "size_unbiased", False):
            # Opt-in path. Returning before `rng` is touched below is what keeps the
            # default path byte-identical: with the flag off, control never enters the
            # new code and RNG consumption is unchanged.
            yield from self._iter_size_unbiased()
            return
        rng = random.Random(self.seed + self._call_count)
        self._call_count += 1
        domain_names = self._active_domain_names
        domain_idx_lists = self._domain_idx_lists
        for _ in range(self.steps):
            batch, total_nodes = [], 0
            min_graphs = max(1, int(getattr(self, "min_graphs", 1)))
            while True:
                domain = rng.choice(domain_names)
                idx = rng.choice(domain_idx_lists[domain])
                n = self.n_nodes[idx]
                if total_nodes + n > self.max_nodes:
                    if len(batch) == 0:
                        # Empty batch: skip this oversized graph and keep drawing
                        continue
                    if len(batch) >= min_graphs:
                        # Budget reached — yield current batch without this graph
                        break
                    # Below min_graphs: accept the overrun rather than emit a batch too
                    # small for the model. Even with MIXED domains a batch of 1 occurs
                    # whenever two large graphs are drawn back to back (two sbm graphs =
                    # 374 nodes vs a 256 budget, ~4% of batches), and a model with
                    # BatchNorm (Pard) raises on it. `min_graphs_per_batch` was previously
                    # stored and NEVER READ, so it silently did nothing.
                batch.append(idx)
                total_nodes += n
                if total_nodes >= self.max_nodes and len(batch) >= min_graphs:
                    break
            yield batch


class _RankStridedBatchSampler(torch.utils.data.Sampler):
    """Rank-r view of an UNMODIFIED batch sampler: yields ``batch[rank::world_size]``.

    This is how `train.max_nodes_per_batch` becomes a GLOBAL node budget under DDP.

    The inner sampler's ``__iter__`` runs VERBATIM on every rank -- same seed
    (``random.Random(self.seed + self._call_count)``), same draw order, same
    accept/reject/carry accounting -- so every rank materializes the IDENTICAL global
    batch and then keeps only its stride slice. The union over ranks is therefore exactly
    the batch a 1-GPU run at the same ``max_nodes_per_batch`` would have trained on. That
    set equality is the parity contract, and it is what makes an N-GPU run comparable to a
    1-GPU run rather than merely statistically similar.

    Why re-run the draw loop on every rank instead of sharding the stream itself: the loop
    is pure Python over a precomputed ``n_nodes`` list (no tensor work, no IO), so the
    duplicated cost is microseconds per batch, and it is what makes the shards EXACT. It
    also covers both the default ``__iter__`` and the opt-in ``_iter_size_unbiased`` carry
    path with no per-path code, because this wrapper only consumes what the inner yields.

    ``__len__`` is the inner's, so every rank yields the same number of batches and no rank
    can run out early (which would hang the others in the gradient all-reduce), and the
    optimizer-steps-per-epoch count is unchanged from a 1-GPU run.

    NOTE the partition is by POSITION, which balances graph COUNT but not cost: dense E is
    O(B_r * N_max_r^2), so a rank that draws the one large graph pads its whole batch and
    stalls the others at the all-reduce. Any deterministic partition of the same global
    batch preserves the contract above (it is a set-equality property, and the
    count-weighted loss scaling makes the result independent of how the split was made),
    so this can be replaced by a cost-balanced partition later. Do NOT replace it with a
    size-SORTED one: in this corpus graph size is nearly a proxy for domain (zinc ~23
    nodes, planar/tree 64, sbm up to 187), so sorting by size would sort by domain and
    starve the domain-specific heads on most ranks.
    """

    def __init__(self, inner, rank: int, world_size: int, n_nodes=None, balanced: bool = True):
        super().__init__(None)
        self.inner = inner
        self.rank = int(rank)
        self.world_size = int(world_size)
        # Node counts enable BALANCED partitioning (see _balanced_shard). Without them the
        # sampler falls back to the positional stride.
        self.n_nodes = n_nodes
        self.balanced = bool(balanced) and n_nodes is not None

    def _balanced_shard(self, batch, step):
        """Greedy largest-first bin-packing by node count; returns THIS rank's bin.

        WHY: positional striding balances the GRAPH COUNT but not the NODE count, and
        per-rank memory tracks nodes. Measured on the real corpus (budget 2048, ws=4):

            positional stride : mean 502, MAX 804 nodes/rank (max/mean 1.60x)
            balanced          : mean 502, MAX 524 nodes/rank (max/mean 1.04x)

        At ~34 GB per 512 nodes/rank that is a 53.4 GB peak vs 34.8 GB -- the difference
        between OOMing on a 47.5 GB card and not. The failure mode it fixes is nasty:
        training survives for hours on typical batches and then dies when it draws from
        the tail (observed at epoch 6227 of a resumed run).

        STILL AN EXACT PARTITION, so nothing about correctness changes:
          * the union over ranks is byte-for-byte the same multiset as the 1-GPU batch,
            which is what Gate B1 asserts;
          * the GRADIENT is unchanged -- Gate C-prime (T4) proves the count-weighted loss
            scaling reconstructs the full-batch gradient from ANY partition, so which rank
            holds which graph is numerically irrelevant.
        So this is a pure memory win, not a tradeoff.

        Deterministic across ranks: the sort is stable (ties keep batch order) and the
        bin choice breaks ties on index, so every rank computes the identical partition
        without communicating.
        """
        order = sorted(batch, key=lambda i: -self.n_nodes[i])
        bins = [[] for _ in range(self.world_size)]
        load = [0] * self.world_size
        for i in order:
            b = min(range(self.world_size), key=lambda k: (load[k], k))
            bins[b].append(i)
            load[b] += self.n_nodes[i]
        # Rotate which bin this rank takes, for the same reason the positional stride
        # rotates: it keeps any residual positional bias from settling on one rank.
        return bins[(self.rank + step) % self.world_size]

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        for step, batch in enumerate(self.inner):
            # Rotate the stride offset per batch instead of using a fixed `rank::ws`.
            #
            # A FIXED offset ties a rank to a POSITION, and position is not neutral. In
            # `_iter_size_unbiased` the carried graph is always prepended at position 0
            # (`batch.append(carry)` before the fill loop), and a carried graph is by
            # construction one that did NOT fit the previous budget -- i.e. a large one.
            # With a fixed offset, rank 0 therefore collects the oversized graphs on every
            # carry, and since graph size is nearly a proxy for domain here (zinc ~23
            # nodes, planar/tree 64, sbm up to 187) that is a systematic rank<->domain
            # correlation -- the MoE expert-imbalance failure mode.
            #
            # Measured on a corpus mirroring the real domain/size structure, ws=8, 2048-node
            # global budget, worst per-rank per-domain deviation over an epoch:
            #     fixed offset,   size_unbiased=False:  29.2% -> 24.1% -> 17.8%  (60/250/1000
            #                     steps; shrinking, i.e. ordinary sampling noise)
            #     fixed offset,   size_unbiased=True:   56.2% -> 50.4% -> 38.3%  (persistent,
            #                     and always on sbm -- the signature of the carry bias)
            # Rotation removes the positional coupling and both settings behave alike.
            #
            # `step` advances identically on every rank (all ranks consume the same inner
            # stream), so the offsets stay a partition: the union over ranks is still
            # exactly the global batch, and Gate B1's multiset equality is unaffected.
            if self.balanced:
                shard = self._balanced_shard(batch, step)
            else:
                offset = (self.rank + step) % self.world_size
                shard = batch[offset :: self.world_size]
            if not shard:
                # Unreachable given the min_graphs_per_batch >= world_size check in the
                # loader builders, but a silently empty shard would reach the no-edge
                # early return in training_step and deadlock every other rank, so fail
                # loudly instead of hanging.
                raise RuntimeError(
                    f"[ddp] empty shard on rank {self.rank}: the global batch held "
                    f"{len(batch)} graphs but world_size is {self.world_size}. Raise "
                    f"train.min_graphs_per_batch to at least {self.world_size}."
                )
            yield shard


class SequentialNodeBudgetSampler(torch.utils.data.Sampler):
    """Sequential batch sampler using pre-computed node counts.

    Groups consecutive samples into variable-size batches without loading any
    data objects.  Used for val/test to avoid the O(N) dataset scan that
    DynamicBatchSampler requires.
    """

    def __init__(self, n_nodes, max_nodes_per_batch):
        super().__init__(None)
        self.n_nodes = n_nodes
        self.max_nodes = int(max_nodes_per_batch)
        self._cached_len = self._compute_len()

    def _compute_len(self):
        batches, total = 0, 0
        for n in self.n_nodes:
            if total > 0 and total + n > self.max_nodes:
                batches += 1
                total = n
            else:
                total += n
        return batches + (1 if total > 0 else 0)

    def __len__(self):
        return self._cached_len

    def __iter__(self):
        batch, total_nodes = [], 0
        for idx, n in enumerate(self.n_nodes):
            if len(batch) > 0 and total_nodes + n > self.max_nodes:
                yield batch
                batch, total_nodes = [], 0
            batch.append(idx)
            total_nodes += n
        if batch:
            yield batch


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=cfg.train.batch_size if "debug" not in cfg.general.name else 2,
            num_workers=cfg.train.num_workers,
            pin_memory=getattr(cfg.dataset, "pin_memory", False),
        )
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None
        self._loader_cache = {}
        self._domain_meta_cache = {}
        print(
            f'This dataset contains {len(datasets["train"])} training graphs, {len(datasets["val"])} validation graphs, {len(datasets["test"])} test graphs.'
        )

    def _dataset_key(self, dataset):
        return id(dataset)

    def _get_cached_domain_meta(self, dataset):
        cache_key = self._dataset_key(dataset)
        cached = self._domain_meta_cache.get(cache_key)
        if cached is not None:
            return cached

        meta = _extract_domain_meta_from_dataset(dataset)
        self._domain_meta_cache[cache_key] = meta
        return meta

    def _is_multi_domain_dataset(self, dataset):
        names = self._get_cached_domain_meta(dataset)["domain_names"]
        return isinstance(names, list) and len(names) > 1

    def _use_node_budget_sampling(self, dataset):
        train_cfg = getattr(self.cfg, "train", None)
        if train_cfg is None:
            return False
        mode = str(getattr(train_cfg, "batching_mode", "")).strip().lower()
        if mode != "node_budget":
            return False
        return self._is_multi_domain_dataset(dataset)

    def _build_flat_index_eval_loader(self, dataset):
        batch_size = self.cfg.train.batch_size if "debug" not in self.cfg.general.name else 2
        num_workers = int(self.cfg.train.num_workers)
        persistent_workers = bool(getattr(self.cfg.train, "persistent_workers", False)) and num_workers > 0
        prefetch_factor = int(getattr(self.cfg.train, "prefetch_factor", 2)) if num_workers > 0 else None
        sampler = SequentialSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def _maybe_shard_batch_sampler(self, batch_sampler, min_graphs, mode):
        """Wrap `batch_sampler` in the rank-strided view when running under DDP.

        At world size 1 this returns the sampler UNCHANGED -- the wrapper is never even
        constructed -- so the single-GPU loader is byte-identical to the pre-DDP code.
        """
        rank, world_size = ddp_topology(self.cfg)
        if world_size <= 1:
            return batch_sampler
        if int(min_graphs) < world_size:
            raise ValueError(
                f"[ddp] train.min_graphs_per_batch={min_graphs} < world_size={world_size} "
                f"(batching_mode={mode}). Stride sharding gives rank r the graphs "
                f"batch[r::{world_size}], so a global batch with fewer than {world_size} "
                f"graphs leaves some rank empty, which deadlocks the gradient all-reduce. "
                f"Set train.min_graphs_per_batch >= {world_size}, and raise "
                f"train.max_nodes_per_batch alongside it so the node budget -- not the "
                f"graph floor -- stays the binding constraint (at max_nodes_per_batch=256 "
                f"the mean batch is only ~4-12 graphs)."
            )
        print(
            f"[ddp] rank {rank}/{world_size}: sharding {mode} batches by stride; "
            f"train.max_nodes_per_batch is the GLOBAL node budget",
            flush=True,
        )
        return _RankStridedBatchSampler(
            batch_sampler, rank, world_size,
            n_nodes=getattr(batch_sampler, "n_nodes", None),
            balanced=bool(getattr(self.cfg.train, "ddp_balanced_shards", True)),
        )

    def _build_node_budget_loader(self, dataset):
        domain_meta = self._get_cached_domain_meta(dataset)
        n_nodes = _extract_node_counts_from_dataset(dataset)
        if n_nodes is None:
            raise RuntimeError("[node_budget] batching_mode=node_budget needs n_nodes_by_global_idx on the dataset")
        max_nodes = int(getattr(self.cfg.train, "max_nodes_per_batch", 512))
        min_graphs = int(getattr(self.cfg.train, "min_graphs_per_batch", 1))
        steps = int(getattr(self.cfg.train, "steps_per_epoch", 250))
        seed = int(getattr(self.cfg.train, "seed", 0))
        size_unbiased = bool(getattr(self.cfg.train, "node_budget_size_unbiased", False))
        draw_w = getattr(self.cfg.train, "domain_draw_weights", None)
        draw_w = dict(draw_w) if draw_w else None
        batch_sampler = NodeBudgetWeightedBatchSampler(
            domain_indices=domain_meta["domain_indices"],
            n_nodes=n_nodes,
            max_nodes_per_batch=max_nodes,
            min_graphs_per_batch=min_graphs,
            steps_per_epoch=steps,
            seed=seed,
            size_unbiased=size_unbiased,
            domain_draw_weights=draw_w,
        )
        batch_sampler = self._maybe_shard_batch_sampler(
            batch_sampler, min_graphs, "node_budget"
        )
        num_workers = int(self.cfg.train.num_workers)
        persistent_workers = bool(getattr(self.cfg.train, "persistent_workers", False)) and num_workers > 0
        prefetch_factor = int(getattr(self.cfg.train, "prefetch_factor", 2)) if num_workers > 0 else None
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def _build_dynamic_eval_loader(self, dataset):
        max_nodes = int(getattr(self.cfg.train, "max_nodes_per_batch", 512))
        n_nodes = _extract_node_counts_from_dataset(dataset)
        if n_nodes is None:
            # Pre-computed node counts unavailable; fall back to fixed-batch sequential loader
            return self._build_flat_index_eval_loader(dataset)
        num_workers = int(self.cfg.train.num_workers)
        persistent_workers = bool(getattr(self.cfg.train, "persistent_workers", False)) and num_workers > 0
        prefetch_factor = int(getattr(self.cfg.train, "prefetch_factor", 2)) if num_workers > 0 else None
        return DataLoader(
            dataset,
            batch_sampler=SequentialNodeBudgetSampler(n_nodes, max_nodes),
            num_workers=num_workers,
            pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def train_dataloader(self):
        if "train" in self._loader_cache:
            return self._loader_cache["train"]

        if self._use_node_budget_sampling(self.train_dataset):
            self._loader_cache["train"] = self._build_node_budget_loader(self.train_dataset)
            return self._loader_cache["train"]
        _num_workers = int(self.cfg.train.num_workers)
        _persistent_workers = bool(getattr(self.cfg.train, "persistent_workers", False)) and _num_workers > 0
        _prefetch_factor = int(getattr(self.cfg.train, "prefetch_factor", 2)) if _num_workers > 0 else None
        self._loader_cache["train"] = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.train.batch_size if "debug" not in self.cfg.general.name else 2,
                shuffle=True,
                num_workers=_num_workers,
                pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
                persistent_workers=_persistent_workers,
                prefetch_factor=_prefetch_factor,
            )
        return self._loader_cache["train"]

    def val_dataloader(self):
        if "val" in self._loader_cache:
            return self._loader_cache["val"]

        _max_nodes = int(getattr(getattr(self.cfg, "train", None), "max_nodes_per_batch", 0) or 0)
        if _max_nodes > 0:
            self._loader_cache["val"] = self._build_dynamic_eval_loader(self.val_dataset)
        else:
            self._loader_cache["val"] = self._build_flat_index_eval_loader(self.val_dataset)
        return self._loader_cache["val"]

    def test_dataloader(self):
        if "test" in self._loader_cache:
            return self._loader_cache["test"]

        _max_nodes = int(getattr(getattr(self.cfg, "train", None), "max_nodes_per_batch", 0) or 0)
        if _max_nodes > 0:
            self._loader_cache["test"] = self._build_dynamic_eval_loader(self.test_dataset)
        else:
            self._loader_cache["test"] = self._build_flat_index_eval_loader(self.test_dataset)
        return self._loader_cache["test"]

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_counts(self, max_nodes_possible=1000):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None

        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes, zinc=False):
        valencies = torch.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])
        if zinc:
            multiplier = torch.tensor([0, 1, 2, 3])  # zinc250

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )

        # ex_dense.E = ex_dense.E[..., :-1]  # debug

        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }

        # unified semantic_global_v1 keeps legacy one-hot fields empty in batches while
        # model-space categorical widths are defined by the global vocab.
        cfg = getattr(datamodule, "cfg", None)
        ds_cfg = getattr(cfg, "dataset", None)
        loader_backend = str(getattr(ds_cfg, "loader_backend", "")).strip().lower()
        unified_cfg = getattr(ds_cfg, "unified", None)
        feature_schema = str(getattr(unified_cfg, "feature_schema", "padded_onehot_v1")).strip().lower()
        is_tokenized_unified = loader_backend == "unified" and feature_schema == "semantic_global_v1"

        global_vocab = getattr(datamodule, "global_vocab", None)
        if is_tokenized_unified and isinstance(global_vocab, dict):
            base_x_dim = int(global_vocab.get("Sx", int(example_batch["x"].size(1))))
            base_e_dim = int(global_vocab.get("Se", int(example_batch["edge_attr"].size(1))))
        else:
            base_x_dim = int(example_batch["x"].size(1))
            base_e_dim = int(example_batch["edge_attr"].size(1))

        self.input_dims = {
            "X": base_x_dim,
            "E": base_e_dim,
            "y": example_batch["y"].size(1)
            + 1,  # this part take into account the conditioning
        }  # + 1 due to time conditioning
        _gen_cfg = getattr(getattr(datamodule, "cfg", None), "general", None)
        # The learned domain embedding replaces the raw one-hot inside the model, widening y.
        _use_expanded_domain_emb = bool(getattr(_gen_cfg, "domain_embedding", False))
        _direct_emb_dim = int(getattr(_gen_cfg, "domain_emb_dim", 0) or 0)
        _n_raw_domains = int(example_batch["y"].size(1))
        if _use_expanded_domain_emb and _direct_emb_dim > _n_raw_domains:
            self.input_dims["y"] += (_direct_emb_dim - _n_raw_domains)
        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_feat.y.size(-1)

        use_domain_onehot = bool(
            getattr(getattr(datamodule, "cfg", None), "general", None) is not None
            and getattr(datamodule.cfg.general, "conditional_domain", False)
        )
        ex_extra_molecular_feat = domain_features(example_data)
        self.mol_input_dims = {
            "X": int(ex_extra_molecular_feat.X.size(-1)),
            "E": int(ex_extra_molecular_feat.E.size(-1)),
            "y": int(ex_extra_molecular_feat.y.size(-1)),
        }
        has_domain_channels = (
            ex_extra_molecular_feat.X.size(-1) > 0
            or ex_extra_molecular_feat.E.size(-1) > 0
            or ex_extra_molecular_feat.y.size(-1) > 0
        )
        if use_domain_onehot and has_domain_channels:
            self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
            self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
            self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            "X": base_x_dim,
            "E": base_e_dim,
            "y": 0,
        }

    def compute_reference_metrics(self, datamodule, sampling_metrics):
        dataset_cfg = getattr(getattr(datamodule, "cfg", None), "dataset", None)
        skip_reference_metrics = bool(
            getattr(dataset_cfg, "skip_reference_metrics", False)
        )

        # Configurable bypass for any dataset where reference MMD metrics should never run.
        if skip_reference_metrics:
            print("Skipping reference metrics computation (dataset.skip_reference_metrics == true).")
            self.ref_metrics = {"val": {}, "test": {}}
            return

        def _extract_domain_local_refs(metrics_obj):
            refs = getattr(metrics_obj, "ref_metrics_by_domain", None)
            if isinstance(refs, Mapping) and len(refs) > 0:
                return refs
            synthetic_eval = getattr(metrics_obj, "synthetic_evaluator", None)
            refs = getattr(synthetic_eval, "ref_metrics_by_domain", None)
            if isinstance(refs, Mapping) and len(refs) > 0:
                return refs
            return None

        domain_local_refs = _extract_domain_local_refs(sampling_metrics)
        if domain_local_refs is not None:
            print("Building dataset_info.ref_metrics from domain-local reference caches.")
            built = {"val": {}, "test": {}}
            for domain_name, split_refs in domain_local_refs.items():
                domain = str(domain_name).strip().lower()
                if not isinstance(split_refs, Mapping):
                    continue
                for split in ["val", "test"]:
                    metric_block = split_refs.get(split)
                    if not isinstance(metric_block, Mapping):
                        continue
                    built[split][domain] = dict(metric_block)
                    for metric_name, metric_value in metric_block.items():
                        built[split][f"cond/domain/{domain}/{metric_name}"] = metric_value
            self.ref_metrics = built
            print("Validation reference metrics:", self.ref_metrics["val"])
            print("Test reference metrics:", self.ref_metrics["test"])
            return

        ref_metrics_path = os.path.join(
            datamodule.train_dataloader().dataset.root, f"ref_metrics.pkl"
        )
        if hasattr(datamodule, "remove_h"):
            if datamodule.remove_h:
                ref_metrics_path = ref_metrics_path.replace(".pkl", "_no_h.pkl")
            else:
                ref_metrics_path = ref_metrics_path.replace(".pkl", "_h.pkl")

        # Only compute the reference metrics if they haven't been computed already
        if not os.path.exists(ref_metrics_path):

            print("Reference metrics not found. Computing them now.")
            # Transform the training dataset into a list of graphs in the appropriate format
            training_graphs = []
            print("Converting training dataset to format required by sampling metrics.")
            _ref_loader = datamodule._build_flat_index_eval_loader(datamodule.train_dataset)
            for data_batch in tqdm(_ref_loader):
                dense_data, node_mask = utils.to_dense(
                    data_batch.x,
                    data_batch.edge_index,
                    data_batch.edge_attr,
                    data_batch.batch,
                )
                dense_data = dense_data.mask(node_mask, collapse=True).split(node_mask)
                for graph in dense_data:
                    training_graphs.append([graph.X, graph.E])

            # defining dummy arguments
            dummy_kwargs = {
                "name": "ref_metrics",
                "current_epoch": 0,
                "val_counter": 0,
                "local_rank": 0,
                "ref_metrics": {"val": None, "test": None},
            }

            print("Computing validation reference metrics.")
            # do not have to worry about wandb because it was not init yet
            val_sampling_metrics = copy.deepcopy(sampling_metrics)

            val_ref_metrics = val_sampling_metrics.forward(
                training_graphs,
                test=False,
                **dummy_kwargs,
            )

            print("Computing test reference metrics.")
            test_sampling_metrics = copy.deepcopy(sampling_metrics)
            test_ref_metrics = test_sampling_metrics.forward(
                training_graphs,
                test=True,
                **dummy_kwargs,
            )

            print("Saving reference metrics.")
            # print(f"deg: {test_reference_metrics['degree']} | clus: {test_reference_metrics['clustering']} | orbit: {test_reference_metrics['orbit']}")
            # breakpoint()
            save_pickle(
                {"val": val_ref_metrics, "test": test_ref_metrics}, ref_metrics_path
            )

        print("Loading reference metrics.")
        self.ref_metrics = load_pickle(ref_metrics_path)
        print("Validation reference metrics:", self.ref_metrics["val"])
        print("Test reference metrics:", self.ref_metrics["test"])
