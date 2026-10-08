"""Score saved generated graphs under the literature protocol, next to DiG2's strict instruments.

    python -m dig2.tools.lit_protocol run  --run-dir <eval run dir> --domain sbm
    python -m dig2.tools.lit_protocol data --domain sbm --split test
    python -m dig2.tools.lit_protocol data --domain qm9 --split train --cfg-run <run dir>

`run` reads the samples file an evaluation run writes (`generated_samples_rank0.pkl`, from
`python -m dig2.train +experiment=eval`); `data` scores a real split the same way, which gives
the ceiling a generator can reach on each instrument. Nothing is generated here.

Published baselines report validity with instruments that differ from one another (isolated
nodes kept or dropped, different SBM refinement budgets, relaxed or plain molecular validity),
so every variant is reported side by side, per graph and pooled:

  structural (sbm / planar / tree)
    strict      isolated nodes dropped; sbm: seeded block-model fit with an ELEMENTWISE Wald test
                (2-5 blocks of 20-40 nodes, mean p > 0.9); planar / tree additionally require the
                graph to have no isolated node, all n generated nodes active, and be connected --
                otherwise a smaller planar/tree subgraph would pass.
    lit_keep / lit_drop
                spectre_utils.is_planar_graph / nx.is_tree / spectre_utils.is_sbm_graph (the
                np.matrix Wald test of the SPECTRE / DiGress code, unchanged) on the graph with
                isolated nodes KEPT (upstream: nx.from_numpy_array of the full n x n adjacency) and
                DROPPED. sbm at refinement_steps 100 (DiGress sbm_acc) and 1000 (the is_sbm_graph
                default inside V.U.N.), suffix _r<steps>.
    V.U.N.      spectre_utils.eval_fraction_unique_non_isomorphic_valid once per validity variant
                (the first member of an isomorphism class counts as unique; novel = not isomorphic
                to any training graph), pooled and per consecutive --block graphs (the
                literature's 5 x 40 protocol uses blocks of 40).
    Every block-model fit is preceded by graph_tool.seed_rng(--gt-seed) and runs on one OpenMP
    thread: the fit is stochastic and its result depends on the thread count, so without both a
    rescore of the same graphs is not reproducible.

  molecular (qm9 / zinc), per molecule, decoded in the run's own vocabulary
    panel_valid RDKit DetectChemistryProblems on the unsanitized molecule, with the N4 / C5 /
                hypervalent-S / other breakdown and fragment count.
    plain       BasicMolecularMetrics.compute_validity per molecule (fragments allowed).
    relaxed     BasicMolecularMetrics.compute_relaxed_validity (+1 formal charge on N/O/S at
                valence+1, full sanitization, largest fragment) -- DiGress / DeFoG "relaxed".
    relaxed_gdss the same partial-charge molecule checked with SANITIZE_PROPERTIES only
                (check_valency) -- GDSS's "validity w/o correction".
    strict      plain-valid and a single fragment.  strict_panel: panel-valid and one fragment.
    uniqueness / novelty over the relaxed-valid (and plain-valid) largest-fragment SMILES;
    novelty against the RDKit-canonical training SMILES.

When the run directory holds the run's own `test_epoch*_res*.txt` / `val_epoch*_res*.txt`, the
molecular plain / relaxed rates are compared to its `Validity` / `Relaxed Validity` (they must
agree to the file's printed precision); for structural domains the file's *_acc values are
copied next to the result for reference only (sampling-time scoring is unseeded).
"""
import argparse
import collections
import contextlib
import glob as _glob
import io
import json
import os
import re
import sys
import time

import numpy as np

from dig2 import paths
from dig2.tools._scoring import (
    GT_SEED, canonical_train_smiles, dense_tokens, domain_items, load_samples, molecular_remaps,
    panel_molecule_row, plain_molecule_verdict, read_run_config, samples_schema,
    semantic_remaps_for_domain, strict_structural_verdict, wilson)

STRUCTURAL = ("sbm", "planar", "tree")
MOLECULAR = ("qm9", "zinc")
DEFAULT_SBM_STEPS = (100, 1000)
DEFAULT_PKL = "generated_samples_rank0.pkl"


# =============================================================================== worker plumbing

def _init_worker():
    """One OpenMP / BLAS thread per process: graph-tool's seeded fit depends on the thread count."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import graph_tool.all as gt
        gt.openmp_set_num_threads(1)
    except Exception:  # noqa: BLE001  (graph-tool is only needed for sbm)
        pass
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _pool_map(fn, jobs, workers, initializer=_init_worker, initargs=(), chunksize=1):
    """spawn, not fork: graph-tool's OpenMP runtime deadlocks in forked children."""
    if workers <= 1 or len(jobs) <= 1:
        initializer(*initargs)
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers, initializer=initializer, initargs=initargs) as pool:
        return pool.map(fn, jobs, chunksize=chunksize)


def _wilson(k, n):
    p, lo, hi = wilson(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": p, "wilson95": [lo, hi]}


def _blocks(n, block):
    """Consecutive FULL blocks only; a trailing partial block is reported, not averaged in."""
    if not block or block <= 0:
        return [], None
    full = [(lo, lo + block) for lo in range(0, n - block + 1, block)]
    tail = (full[-1][1] if full else 0, n) if (n % block) else None
    return full, tail


def _block_stats(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    return {"n_blocks": int(v.size), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "std_pop": float(v.std(ddof=0)), "values": [float(x) for x in v]}


def _rate_block(flags, blocks):
    f = np.asarray(flags, dtype=float)
    return _block_stats([f[lo:hi].mean() for lo, hi in blocks])


# =============================================================================== structural

def _graphs_from_adj(E):
    """(kept, dropped, n_isolated, n_selfloops). kept = the upstream construction (full n x n
    adjacency, self-loops removed); dropped = isolated nodes removed."""
    import networkx as nx
    E = np.asarray(E)
    G = nx.from_numpy_array((E > 0).astype(np.uint8))
    n_self = int(nx.number_of_selfloops(G))
    iso = [v for v in G.nodes if G.degree(v) == 0]
    Ga = G.copy()
    Ga.remove_nodes_from(iso)
    Gk = G.copy()
    Gk.remove_edges_from(list(nx.selfloop_edges(Gk)))
    return Gk, Ga, len(iso), n_self


def _structural_row(job):
    idx, E, domain, sbm_steps, gt_seed = job
    import networkx as nx
    import graph_tool.all as gt
    from dig2.metrics.spectre_utils import is_planar_graph, is_sbm_graph
    n_forced = int(np.asarray(E).shape[0])
    Gk, Ga, n_iso, n_self = _graphs_from_adj(E)
    t0 = time.time()
    gt.seed_rng(gt_seed)
    r = strict_structural_verdict(Ga, domain)
    crit = bool(r["ok"])
    size_ok = (n_iso == 0) and Ga.number_of_nodes() == n_forced and \
        Ga.number_of_nodes() > 0 and nx.is_connected(Ga)
    row = {"i": int(idx), "n_forced": n_forced, "n_active": int(Ga.number_of_nodes()),
           "n_iso": int(n_iso), "n_selfloops": n_self, "n_edges": int(Gk.number_of_edges()),
           "criterion": crit, "size_ok": bool(size_ok),
           "strict": bool(crit and (size_ok if domain in ("planar", "tree") else True)),
           "fail_reason": None if crit else r["fail_reason"],
           "n_blocks": r["n_blocks"], "block_sizes": r["block_sizes"], "wald_p": r["wald_p"]}
    if domain == "planar":
        row["lit_keep"] = bool(Gk.number_of_nodes() > 0 and is_planar_graph(Gk))
        row["lit_drop"] = bool(Ga.number_of_nodes() > 0 and is_planar_graph(Ga))
    elif domain == "tree":
        row["lit_keep"] = bool(Gk.number_of_nodes() > 0 and nx.is_tree(Gk))
        row["lit_drop"] = bool(Ga.number_of_nodes() > 0 and nx.is_tree(Ga))
    elif domain == "sbm":
        for steps in sbm_steps:
            for mode, G in (("drop", Ga), ("keep", Gk)):
                gt.seed_rng(gt_seed)
                row[f"lit_{mode}_r{steps}"] = bool(is_sbm_graph(G, refinement_steps=int(steps)))
    row["sec"] = round(time.time() - t0, 3)
    return row


_VUN_CTX = {}


def _vun_init(graph_sets):
    _init_worker()
    _VUN_CTX.clear()
    _VUN_CTX.update(graph_sets)


def _vun_task(job):
    key, set_key, lo, hi, validity = job
    from dig2.metrics.spectre_utils import eval_fraction_unique_non_isomorphic_valid
    fake = _VUN_CTX[set_key][lo:hi]
    train = _VUN_CTX["train_" + set_key]
    u, un, vun = eval_fraction_unique_non_isomorphic_valid(fake, train, validity_by_index=list(validity))
    return {"key": key, "lo": lo, "hi": hi, "unique": u, "unique_novel": un, "vun": vun}


def _structural_validity_keys(domain, sbm_steps):
    """validity key -> the graph construction its V.U.N. is computed on."""
    keys = {"strict": "drop"}
    if domain in ("planar", "tree"):
        keys.update(lit_keep="keep", lit_drop="drop")
    else:
        for s in sbm_steps:
            keys.update({f"lit_drop_r{s}": "drop", f"lit_keep_r{s}": "keep"})
    return keys


def _raw_split_path(data_dir, domain, split):
    return os.path.join(data_dir, domain, "raw", f"{split}.pt")


def _load_train_graphs(data_dir, domain):
    import torch
    out = {"keep": [], "drop": []}
    for a in torch.load(_raw_split_path(data_dir, domain, "train"), weights_only=False):
        Gk, Ga, _, _ = _graphs_from_adj(a.numpy())
        out["keep"].append(Gk)
        out["drop"].append(Ga)
    return out


def score_structural_items(adjs, domain, block, workers, data_dir, sbm_steps=DEFAULT_SBM_STEPS, vun=True,
                           gt_seed=GT_SEED):
    t0 = time.time()
    jobs = [(i, E, domain, tuple(sbm_steps), int(gt_seed)) for i, E in enumerate(adjs)]
    rows = _pool_map(_structural_row, jobs, workers, chunksize=1)
    rows.sort(key=lambda r: r["i"])
    n = len(rows)
    t_fit = time.time() - t0
    keys = _structural_validity_keys(domain, sbm_steps)
    blocks, tail = _blocks(n, block)
    out = {"n": n, "domain": domain, "gt_seed": int(gt_seed), "omp_threads": 1,
           "block": block, "n_full_blocks": len(blocks), "trailing_partial_block": tail,
           "sbm_refinement_steps": list(sbm_steps) if domain == "sbm" else None,
           "validity": {}, "vun": {}, "diagnostics": {}}
    for k in keys:
        flags = [bool(r[k]) for r in rows]
        out["validity"][k] = {**_wilson(sum(flags), n), "blocks": _rate_block(flags, blocks)}
    for extra in ("criterion", "size_ok"):
        flags = [bool(r[extra]) for r in rows]
        out["diagnostics"][f"{extra}_rate"] = sum(flags) / max(n, 1)
    out["diagnostics"]["n_graphs_with_isolated"] = sum(1 for r in rows if r["n_iso"] > 0)
    out["diagnostics"]["n_graphs_with_selfloops"] = sum(1 for r in rows if r["n_selfloops"] > 0)
    out["diagnostics"]["fail_reasons_strict"] = dict(collections.Counter(
        r["fail_reason"] for r in rows if not r["criterion"]))
    if domain == "sbm":
        # With no isolated node and no self-loop the kept and dropped graphs are the same graph,
        # so their seeded verdicts must agree; a nonzero count means the seeding is broken.
        same = [r for r in rows if r["n_iso"] == 0 and r["n_selfloops"] == 0]
        out["diagnostics"]["keep_vs_drop_disagree_on_identical_graphs"] = {
            f"r{s}": sum(1 for r in same if r[f"lit_keep_r{s}"] != r[f"lit_drop_r{s}"]) for s in sbm_steps}
        out["diagnostics"]["lit_vs_strict_flips"] = {
            f"lit_drop_r{s}": {"lit_only": sum(1 for r in rows if r[f"lit_drop_r{s}"] and not r["strict"]),
                               "strict_only": sum(1 for r in rows if r["strict"] and not r[f"lit_drop_r{s}"])}
            for s in sbm_steps}
        with_iso = [r for r in rows if r["n_iso"] > 0]
        out["diagnostics"]["keep_vs_drop_disagree_on_graphs_with_isolated"] = {
            f"r{s}": sum(1 for r in with_iso if r[f"lit_keep_r{s}"] != r[f"lit_drop_r{s}"]) for s in sbm_steps}
        steps = list(sbm_steps)
        out["diagnostics"]["refinement_steps_disagree"] = {
            f"{mode}_r{a}_vs_r{b}": sum(1 for r in rows if r[f"lit_{mode}_r{a}"] != r[f"lit_{mode}_r{b}"])
            for mode in ("drop", "keep") for ai, a in enumerate(steps) for b in steps[ai + 1:]}
    else:
        out["diagnostics"]["strict_vs_lit_keep_disagree"] = sum(1 for r in rows if r["strict"] != r["lit_keep"])
    out["timing_s"] = {"fit": round(t_fit, 1)}
    if vun:
        t1 = time.time()
        sets = {"keep": [], "drop": []}
        for E in adjs:
            Gk, Ga, _, _ = _graphs_from_adj(E)
            sets["keep"].append(Gk)
            sets["drop"].append(Ga)
        tr = _load_train_graphs(data_dir, domain)
        sets["train_keep"], sets["train_drop"] = tr["keep"], tr["drop"]
        vjobs = []
        for k, set_key in keys.items():
            flags = [bool(r[k]) for r in rows]
            vjobs.append((k, set_key, 0, n, flags))
            for lo, hi in blocks:
                vjobs.append((k, set_key, lo, hi, flags[lo:hi]))
        res = _pool_map(_vun_task, vjobs, min(workers, len(vjobs)), initializer=_vun_init, initargs=(sets,))
        for k, set_key in keys.items():
            pooled = [x for x in res if x["key"] == k and x["lo"] == 0 and x["hi"] == n][0]
            blk = [x for x in res if x["key"] == k and not (x["lo"] == 0 and x["hi"] == n)]
            if len(blocks) == 1 and blocks[0] == (0, n):
                blk = [pooled]
            blk = sorted(blk, key=lambda b: b["lo"])
            k_vun = int(round(pooled["vun"] * n))
            out["vun"][k] = {"graphs": set_key, **_wilson(k_vun, n),
                             "unique": pooled["unique"], "unique_novel": pooled["unique_novel"],
                             "blocks": _block_stats([b["vun"] for b in blk]),
                             "blocks_unique": _block_stats([b["unique"] for b in blk]),
                             "blocks_unique_novel": _block_stats([b["unique_novel"] for b in blk])}
        out["n_train_graphs"] = len(sets["train_keep"])
        out["timing_s"]["vun"] = round(time.time() - t1, 1)
    out["rows"] = rows
    return out


# =============================================================================== molecular

_MOL_CTX = {}


def _mol_init(ar, er, dec):
    _init_worker()
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from types import SimpleNamespace
    from dig2.metrics.rdkit_functions import BasicMolecularMetrics
    _MOL_CTX.update(ar=ar, er=er, dec=list(dec),
                    bmm=BasicMolecularMetrics(SimpleNamespace(atom_decoder=list(dec))))


def _mol_row(job):
    idx, X, E = job
    import torch
    from dig2.metrics.rdkit_functions import build_molecule_with_partial_charges, check_valency
    ar, er, dec, bmm = _MOL_CTX["ar"], _MOL_CTX["er"], _MOL_CTX["dec"], _MOL_CTX["bmm"]
    xg = torch.as_tensor(np.asarray(X)).long()
    Eg = torch.as_tensor(np.asarray(E)).long()
    row = {"i": int(idx), "n_atoms": int(xg.numel())}
    pr = panel_molecule_row(xg, Eg, (ar, er, dec))
    row.update(panel_valid=bool(pr["valid"]), n4=int(pr["n4"]), c5=int(pr["c5"]), s_hyper=int(pr["s_hyper"]),
               other=int(pr["other_problem"]), n_frag=int(pr["n_frag"]))
    ev = plain_molecule_verdict(xg, Eg, ar, er, dec)
    row.update(plain=bool(ev["ok"]), plain_fail=ev["fail_reason"], plain_smiles=ev["smiles"],
               plain_n_frags=ev["n_fragments"])
    Xl = ar[xg.clamp(0, ar.numel() - 1)]
    El = er[Eg.clamp(0, er.numel() - 1)]
    # compute_relaxed_validity on a one-molecule list; it prints and draws a progress bar.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        valid_list, rv = bmm.compute_relaxed_validity([(Xl, El.clone())])
    row["relaxed"] = bool(rv == 1.0)
    row["relaxed_smiles"] = valid_list[0] if valid_list else None
    try:
        m = build_molecule_with_partial_charges(Xl, El.clone(), dec)
        row["relaxed_gdss"] = bool(check_valency(m)[0])
    except Exception as ex:  # noqa: BLE001
        row["relaxed_gdss"] = False
        row["gdss_error"] = str(ex)[:200]
    row["strict"] = bool(row["plain"] and row["plain_n_frags"] == 1)
    row["strict_panel"] = bool(row["panel_valid"] and row["n_frag"] == 1)
    return row


def score_molecular_items(items, domain, remaps, block, workers, train_can=None, keep_rows=True):
    t0 = time.time()
    ar, er, dec = remaps
    jobs = [(i, X, E) for i, (X, E) in enumerate(items)]
    rows = _pool_map(_mol_row, jobs, workers, initializer=_mol_init, initargs=(ar, er, dec),
                     chunksize=max(1, min(256, len(jobs) // max(1, 4 * workers))))
    rows.sort(key=lambda r: r["i"])
    n = len(rows)
    blocks, tail = _blocks(n, block)
    out = {"n": n, "domain": domain, "block": block, "n_full_blocks": len(blocks),
           "trailing_partial_block": tail, "atom_decoder": list(dec), "validity": {}, "breakdown": {}}
    for k in ("panel_valid", "plain", "relaxed", "relaxed_gdss", "strict", "strict_panel"):
        flags = [bool(r[k]) for r in rows]
        out["validity"][k] = {**_wilson(sum(flags), n), "blocks": _rate_block(flags, blocks)}
    for k in ("n4", "c5", "s_hyper", "other"):
        out["breakdown"][f"rate_{k}"] = sum(r[k] for r in rows) / max(n, 1)
        out["breakdown"][f"count_{k}"] = sum(r[k] for r in rows)
    nv = sum(r["panel_valid"] for r in rows)
    out["breakdown"]["frag_among_panel_valid"] = sum(1 for r in rows if r["panel_valid"] and r["n_frag"] > 1) / max(nv, 1)
    out["breakdown"]["frag_rate_all"] = sum(1 for r in rows if r["n_frag"] > 1) / max(n, 1)
    out["breakdown"]["plain_fail_reasons"] = dict(collections.Counter(r["plain_fail"] for r in rows if not r["plain"]))
    out["breakdown"]["panel_vs_plain_disagree"] = sum(1 for r in rows if r["panel_valid"] != r["plain"])
    rel_only = [r for r in rows if r["relaxed"] and not r["relaxed_gdss"]]
    out["breakdown"]["relaxed_vs_gdss_disagree"] = {
        "relaxed_only": len(rel_only),
        "gdss_only": sum(1 for r in rows if r["relaxed_gdss"] and not r["relaxed"]),
        # The partial-charge build only sets +1 charges, so a NEGATIVE charge in a relaxed SMILES
        # comes from full sanitization's cleanup (e.g. N(=O)=O -> [N+](=O)[O-]), which GDSS's
        # SANITIZE_PROPERTIES-only check never runs.
        "relaxed_only_with_negative_charge": sum(1 for r in rel_only if "-]" in (r["relaxed_smiles"] or "")),
        "relaxed_only_examples": [r["relaxed_smiles"] for r in rel_only[:12]],
        "gdss_build_errors": sum(1 for r in rows if "gdss_error" in r)}
    rel = [r["relaxed_smiles"] for r in rows if r["relaxed"] and r["relaxed_smiles"] is not None]
    pla = [r["plain_smiles"] for r in rows if r["plain"] and r["plain_smiles"] is not None]
    uq = {"relaxed": rel, "plain": pla}
    out["uniqueness"] = {k: (len(set(v)) / len(v) if v else None) for k, v in uq.items()}
    out["uniqueness_blocks"] = {}
    for k in uq:
        vals = []
        for lo, hi in blocks:
            s = [r[f"{k}_smiles"] for r in rows[lo:hi] if r[k] and r[f"{k}_smiles"] is not None]
            vals.append(len(set(s)) / len(s) if s else float("nan"))
        out["uniqueness_blocks"][k] = _block_stats(vals)
    if train_can is not None:
        out["novelty"] = {k: (sum(1 for s in set(v) if s not in train_can) / len(set(v)) if v else None)
                          for k, v in uq.items()}
        out["n_train_smiles"] = len(train_can)
    out["timing_s"] = round(time.time() - t0, 1)
    if keep_rows:
        out["rows"] = rows
    return out


def read_harness_txt(run_dir, domain):
    """The run's own metrics file (multi-domain keys cond/domain/<d>/<k>, single-domain <k>);
    the test file when present, else the last validation file."""
    txts = sorted(_glob.glob(os.path.join(run_dir, "test_epoch*_res*.txt"))) or \
        sorted(_glob.glob(os.path.join(run_dir, "val_epoch*_res*.txt")))
    if not txts:
        return None
    vals = {}
    with open(txts[-1]) as f:
        for line in f:
            m = re.match(r"^(?:cond/domain/([^/]+)/)?([^:]+):\s*([-0-9.eE+naninf]+)\s*$", line.strip())
            if not m:
                continue
            d, k, v = m.group(1), m.group(2), m.group(3)
            if d is not None and d != domain:
                continue
            try:
                vals[k] = float(v)
            except ValueError:
                pass
    return {"file": txts[-1], "values": vals}


def check_molecular_txt(res, run_dir, domain):
    t = read_harness_txt(run_dir, domain)
    if t is None:
        return None
    out = {"file": t["file"]}
    for ours_k, txt_k in (("plain", "Validity"), ("relaxed", "Relaxed Validity")):
        if txt_k in t["values"]:
            a, b = res["validity"][ours_k]["rate"], t["values"][txt_k]
            out[ours_k] = {"ours": a, "harness": b, "abs_diff": abs(a - b),
                           "equal_to_txt_precision": abs(a - b) < 5e-7 * max(1.0, abs(b)) + 1e-12}
    return out


# =============================================================================== loaders

def _run_vocab(run_dir, samples, vocab):
    """Which vocabulary the run's molecules are in, and remove_h. The multi-domain sampler (dict
    schema) always emits the global semantic vocabulary; a list-schema file is global only if the
    run used the unified loader, which is read from the run's recorded config. With no config
    the decision is refused rather than guessed: decoding in the wrong vocabulary yields
    plausible-looking but wrong molecules (validity near 0)."""
    schema = samples_schema(samples)
    cfg = read_run_config(run_dir)
    ds = (cfg or {}).get("dataset") or {}
    remove_h = bool(ds["remove_h"]) if ds.get("remove_h") is not None else True
    if vocab in ("global", "local"):
        global_vocab, source = vocab == "global", "forced"
    elif schema == "dict":
        global_vocab, source = True, "dict schema"
    elif cfg is not None:
        global_vocab = str(ds.get("loader_backend", "") or "").strip().lower().startswith("unified")
        source = "run config dataset.loader_backend"
    else:
        raise SystemExit(f"{run_dir}: list-schema samples and no .hydra/config.yaml; cannot tell whether "
                         f"the molecules use the global or the domain vocabulary -- pass --vocab")
    return global_vocab, remove_h, {"pkl_schema": schema, "global_vocab": global_vocab,
                                    "vocab_source": source, "remove_h": remove_h}


def _unified_split_items(domain, split, cfg_run, data_dir=None, limit=0):
    """A molecular domain's real graphs exactly as training sees them (global semantic tokens),
    from the unified cache selected by a run's recorded config. Refuses to build a missing cache
    (that takes ~10 minutes and should not happen as a side effect of scoring)."""
    from omegaconf import OmegaConf
    from dig2.data.unified.dataset import UnifiedDataset
    cfg = OmegaConf.load(os.path.join(cfg_run, ".hydra", "config.yaml"))
    OmegaConf.set_struct(cfg, False)
    if data_dir is not None:
        cfg.dataset.datadir = os.path.abspath(data_dir)
    root = os.path.join(str(paths.repo_root()), str(cfg.dataset.datadir),
                        str(getattr(getattr(cfg.dataset, "unified", None), "root_subdir", "unified")))
    if not UnifiedDataset.is_cache_ready(cfg, split, root):
        raise SystemExit(f"unified cache for split={split} under {root} (config {cfg_run}) is not ready; "
                         f"refusing to build it")
    ds = UnifiedDataset(cfg, split, root, bool(getattr(cfg.general, "conditional_domain", False)))
    idx = list(ds.domain_indices[domain])
    if limit:
        idx = idx[:limit]
    items = []
    for gi in idx:
        xg, Eg = dense_tokens(ds[int(gi)])
        items.append((xg.numpy().astype(np.int64), Eg.numpy().astype(np.int64)))
    meta = {"source": "unified", "cfg_run": cfg_run, "cache_root": root,
            "remove_h": bool(cfg.dataset.remove_h), "n_domain_graphs": len(ds.domain_indices[domain])}
    return items, meta, bool(cfg.dataset.remove_h)


# =============================================================================== commands

def _dump(res, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[lit] wrote {out_path}", flush=True)


def _brief(res):
    v = {k: round(x["rate"], 4) for k, x in res["validity"].items()}
    if res["domain"] in STRUCTURAL:
        u = {k: (round(x["rate"], 4), None if not x["blocks"] else
                 f"{x['blocks']['mean']:.4f}+-{x['blocks']['std']:.4f}") for k, x in res.get("vun", {}).items()}
        return f"validity={json.dumps(v)}\n    vun(pooled, block mean+-std)={json.dumps(u)}"
    b = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in res["breakdown"].items()
         if k.startswith("rate_") or k.startswith("frag")}
    return (f"validity={json.dumps(v)}\n    breakdown={json.dumps(b)}\n    uniqueness={json.dumps(res.get('uniqueness'))}"
            f" novelty={json.dumps(res.get('novelty'))}")


def _data_dir(a):
    return os.path.abspath(a.data_dir) if a.data_dir else str(paths.repo_root() / "data")


def _train_smiles_path(a, dom):
    return a.train_smiles or os.path.join(_data_dir(a), dom, "train_smiles_no_h.npy")


def cmd_run(a):
    dom = a.domain.strip().lower()
    run_dir = os.path.abspath(a.run_dir)
    pkl = a.pkl if os.path.isabs(a.pkl) else os.path.join(run_dir, a.pkl)
    samples = load_samples(pkl)
    items = domain_items(samples, dom)
    if a.limit:
        items = items[: a.limit]
    if not items:
        raise SystemExit(f"{pkl}: no {dom} graphs")
    block = a.block if a.block is not None else (40 if dom in STRUCTURAL else 10000)
    meta = {"cmd": "run", "run_dir": run_dir, "pkl": pkl, "argv": sys.argv, "workers": a.workers,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if dom in STRUCTURAL:
        steps = tuple(int(s) for s in a.sbm_steps.split(","))
        res = score_structural_items([E for _X, E in items], dom, block, a.workers, _data_dir(a), sbm_steps=steps,
                                     vun=not a.no_vun, gt_seed=a.gt_seed)
        t = read_harness_txt(run_dir, dom)
        if t is not None:
            res["harness_txt"] = {k: v for k, v in t["values"].items()
                                  if k.endswith("_acc") or k.startswith("sampling/frac")} | {"file": t["file"]}
    elif dom in MOLECULAR:
        global_vocab, remove_h, vmeta = _run_vocab(run_dir, samples, a.vocab)
        meta.update(vmeta)
        remaps = molecular_remaps(dom, global_vocab, remove_h)
        can = None if a.no_novelty else canonical_train_smiles(_train_smiles_path(a, dom))
        res = score_molecular_items(items, dom, remaps, block, a.workers, train_can=can)
        if not a.limit:
            res["harness_txt_check"] = check_molecular_txt(res, run_dir, dom)
    else:
        raise SystemExit(f"unknown domain {dom}")
    res["meta"] = meta
    out = a.out or os.path.join(run_dir, f"lit_{dom}.json")
    _dump(res, out)
    print(f"[lit][run] {os.path.basename(run_dir)} {dom} n={res['n']}\n    {_brief(res)}", flush=True)
    if res.get("harness_txt_check"):
        print(f"    harness_txt_check: {json.dumps({k: v for k, v in res['harness_txt_check'].items() if k != 'file'})}",
              flush=True)


def cmd_data(a):
    dom = a.domain.strip().lower()
    block = a.block if a.block is not None else (40 if dom in STRUCTURAL else 10000)
    meta = {"cmd": "data", "split": a.split, "argv": sys.argv, "workers": a.workers,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if dom in STRUCTURAL:
        import torch
        src = _raw_split_path(_data_dir(a), dom, a.split)
        adjs = [x.numpy() for x in torch.load(src, weights_only=False)]
        if a.limit:
            adjs = adjs[: a.limit]
        meta["source"] = src
        steps = tuple(int(s) for s in a.sbm_steps.split(","))
        res = score_structural_items(adjs, dom, block, a.workers, _data_dir(a), sbm_steps=steps, vun=not a.no_vun,
                                     gt_seed=a.gt_seed)
        if a.split == "train" and "vun" in res:
            res["vun_note"] = "train split vs itself: every graph is isomorphic to a train graph, V.U.N. is 0 by construction"
    elif dom in MOLECULAR:
        if not a.cfg_run:
            raise SystemExit("molecular data splits need --cfg-run <run dir> (its .hydra/config.yaml selects the cache)")
        items, smeta, remove_h = _unified_split_items(dom, a.split, a.cfg_run, data_dir=a.data_dir, limit=a.limit)
        meta.update(smeta)
        remaps = semantic_remaps_for_domain(dom, remove_h=remove_h)
        can = None if a.no_novelty else canonical_train_smiles(_train_smiles_path(a, dom))
        res = score_molecular_items(items, dom, remaps, block, a.workers, train_can=can, keep_rows=a.rows)
    else:
        raise SystemExit(f"unknown domain {dom}")
    res["meta"] = meta
    out = a.out or os.path.abspath(f"lit_data_{dom}_{a.split}.json")
    _dump(res, out)
    print(f"[lit][data] {dom}/{a.split} n={res['n']}\n    {_brief(res)}", flush=True)


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "data"):
        p = sub.add_parser(name)
        p.add_argument("--domain", required=True, choices=STRUCTURAL + MOLECULAR)
        p.add_argument("--block", type=int, default=None,
                       help="consecutive block size for per-block stats (default 40 structural, 10000 molecular)")
        p.add_argument("--out", default=None,
                       help="output json (default: <run-dir>/lit_<domain>.json for run, ./lit_data_<domain>_<split>.json for data)")
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--limit", type=int, default=0, help="first N graphs only (smoke tests)")
        p.add_argument("--sbm-steps", default=",".join(str(s) for s in DEFAULT_SBM_STEPS),
                       help="is_sbm_graph refinement_steps variants (comma list)")
        p.add_argument("--no-vun", action="store_true")
        p.add_argument("--no-novelty", action="store_true", help="skip loading the training SMILES (molecular)")
        p.add_argument("--gt-seed", type=int, default=GT_SEED, help="graph_tool.seed_rng before every fit")
        p.add_argument("--data-dir", default=None,
                       help="dataset root holding <domain>/raw/{train,val,test}.pt and <domain>/train_smiles_no_h.npy "
                            "(default <repo>/data)")
        p.add_argument("--train-smiles", default=None,
                       help="training SMILES .npy for novelty (default <data-dir>/<domain>/train_smiles_no_h.npy)")
    r = sub.choices["run"]
    r.add_argument("--run-dir", required=True, help="an evaluation run directory")
    r.add_argument("--pkl", default=DEFAULT_PKL, help=f"samples file, relative to --run-dir or absolute (default {DEFAULT_PKL})")
    r.add_argument("--vocab", default="auto", choices=("auto", "global", "local"),
                   help="molecular vocabulary of the samples: auto = decided from the file schema / run config")
    d = sub.choices["data"]
    d.add_argument("--split", required=True, choices=("train", "val", "test"))
    d.add_argument("--cfg-run", default=None,
                   help="run dir whose .hydra/config.yaml selects the unified cache (molecular domains)")
    d.add_argument("--rows", action="store_true", help="keep per-molecule rows in the json (large)")
    a = ap.parse_args()
    {"run": cmd_run, "data": cmd_data}[a.cmd](a)


if __name__ == "__main__":
    main()
