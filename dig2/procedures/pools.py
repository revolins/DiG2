"""Fresh structural training pools for anchored fine-tuning, and the fidelity gate they must pass.

SBM and Planar ship with 128 training graphs each. Anchored fine-tuning trains on a pool of graphs drawn from the same generative processes, so target domains are not
memorised during a 25k-step fine-tune. Only the TRAIN split is replaced: val/test are copied
verbatim from the original (SPECTRE) dataset, so every validation and test reference is unchanged.

A pool directory has SpectreGraphDataset's raw layout, which is what
``dataset.unified.domain_root_overrides`` points at:

    <root>/raw/train.pt     list of dense float adjacency matrices (the fresh pool)
    <root>/raw/val.pt       copied from the original dataset's raw/
    <root>/raw/test.pt      copied from the original dataset's raw/
    <root>/fidelity_gate.json

``processed/`` is created by the loader on first use, and the unified cache of an override run
gets its own directory, so an override never reads or clobbers the default cache.

The fidelity gate exists because a pool from a shifted distribution would quietly fine-tune
toward the wrong target. On ``gate_n`` pool graphs against the original graphs it checks:
  * node-count and edge-density KS tests (density matters: the per-domain noise prior is rebuilt
    from TRAINING marginals, so a density shift would make training and evaluation priors disagree);
  * sbm only: the block-model criterion pass rate, Fisher-tested against the original dataset's;
  * degree / clustering / spectral MMD: is the pool as close to the held-out graphs as the
    original training graphs are? See :func:`mmd_null_gate`.
"""
import json
import os
import shutil
import time

import networkx as nx
import numpy as np
import scipy.spatial
import torch

POOL_DOMAINS = ("sbm", "planar")
DEFAULT_N = 8192
DEFAULT_SEED = 20260921
DEFAULT_GATE_N = 2000
DEFAULT_MMD_NULL_B = 200

# Generator parameters of the original datasets: 2-5 communities of 20-40 nodes with p_intra 0.3
# and p_inter 0.005; planar graphs are Delaunay triangulations of 64 uniform points.
SBM_N_COMMUNITIES = (2, 5)
SBM_COMMUNITY_SIZE = (20, 40)
SBM_P_INTRA = 0.3
SBM_P_INTER = 0.005
PLANAR_N_NODES = 64

# Block-model criterion on the original sbm graphs (train 106/128 + val 26/32 + test 32/40),
# measured with sbm_criterion below under graph-tool seed SBM_CRITERION_GT_SEED.
SPECTRE_SBM_PASS = (164, 200)
SBM_CRITERION_GT_SEED = 12345
SBM_WALD_P_THRESHOLD = 0.9
SBM_REFINEMENT_STEPS = 100


# ------------------------------------------------------------------------------------------------
# generators
# ------------------------------------------------------------------------------------------------

def generate_sbm(n_graphs, seed, n_communities=SBM_N_COMMUNITIES, community_size=SBM_COMMUNITY_SIZE,
                 p_intra=SBM_P_INTRA, p_inter=SBM_P_INTER):
    """Connected SBM graphs as dense float adjacencies. Disconnected draws are rejected (the
    original dataset contains only connected graphs), so the output is a deterministic function
    of ``seed`` but the number of RNG draws per kept graph is not fixed."""
    rng = np.random.default_rng(seed)
    adjs = []
    while len(adjs) < n_graphs:
        n_comm = int(rng.integers(n_communities[0], n_communities[1] + 1))
        sizes = [int(rng.integers(community_size[0], community_size[1] + 1)) for _ in range(n_comm)]
        p = [[p_intra if i == j else p_inter for j in range(n_comm)] for i in range(n_comm)]
        g = nx.stochastic_block_model(sizes, p, seed=int(rng.integers(1e9)))
        g.remove_edges_from(nx.selfloop_edges(g))
        if nx.is_connected(g):
            adj = torch.tensor(nx.to_numpy_array(g), dtype=torch.float)
            adj.fill_diagonal_(0.0)
            adjs.append(adj)
    return adjs


def generate_planar(n_graphs, seed, n_nodes=PLANAR_N_NODES):
    """Delaunay triangulations of ``n_nodes`` uniform points in the unit square. The
    connected/planar filter never rejects a proper triangulation; it guards degenerate inputs."""
    rng = np.random.default_rng(seed)
    adjs = []
    while len(adjs) < n_graphs:
        pts = rng.uniform(size=(n_nodes, 2))
        tri = scipy.spatial.Delaunay(pts)
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        for s in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    g.add_edge(int(s[i]), int(s[j]))
        if nx.is_connected(g) and nx.is_planar(g):
            adj = torch.tensor(nx.to_numpy_array(g), dtype=torch.float)
            adj.fill_diagonal_(0.0)
            adjs.append(adj)
    return adjs


def generate(domain, n_graphs, seed):
    if domain == "sbm":
        return generate_sbm(n_graphs, seed)
    if domain == "planar":
        return generate_planar(n_graphs, seed)
    raise ValueError(f"no pool generator for domain {domain!r}; expected one of {POOL_DOMAINS}")


def to_nx(adjs):
    return [nx.from_numpy_array((a.numpy() if torch.is_tensor(a) else np.asarray(a)) > 0) for a in adjs]


# ------------------------------------------------------------------------------------------------
# sbm block-model criterion
# ------------------------------------------------------------------------------------------------

def sbm_criterion(G, p_intra=SBM_P_INTRA, p_inter=SBM_P_INTER, refinement_steps=SBM_REFINEMENT_STEPS):
    """True iff a fitted block model has 2-5 blocks of 20-40 nodes whose densities pass the mean
    Wald test (p > 0.9) against (p_intra, p_inter).

    The fit is the one dig2.metrics.spectre_utils.is_sbm_graph performs, but the Wald statistic is
    computed with element-wise array arithmetic. is_sbm_graph keeps the edge-count matrix as the
    np.matrix returned by ``todense()``, on which ``**`` is a matrix power and ``*`` a matrix
    product, and that inflates its pass rate. SPECTRE_SBM_PASS was measured with this function,
    so the gate compares like with like. graph-tool's MCMC is seeded by the caller."""
    import graph_tool.all as gt
    from scipy.stats import chi2

    try:
        A = nx.to_scipy_sparse_array(G, format="coo")
        edge_index = np.column_stack((A.row, A.col))
        g = gt.Graph()
        g.add_vertex(G.number_of_nodes())
        if edge_index.size > 0:
            g.add_edge_list(edge_index)
        state = gt.minimize_blockmodel_dl(g)
        for _ in range(refinement_steps):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
        state = state.copy(b=gt.contiguous_map(state.get_blocks()))
        n_blocks = state.get_nonempty_B()
        node_counts = state.get_nr().get_array()[:n_blocks]
        edge_counts = np.asarray(state.get_matrix().todense())[:n_blocks, :n_blocks]
    except Exception:
        return False
    if not SBM_N_COMMUNITIES[0] <= n_blocks <= SBM_N_COMMUNITIES[1]:
        return False
    if (node_counts < SBM_COMMUNITY_SIZE[0]).any() or (node_counts > SBM_COMMUNITY_SIZE[1]).any():
        return False
    est_p_intra = np.diagonal(edge_counts) / (node_counts * (node_counts - 1) + 1e-6)
    off = edge_counts.copy()
    np.fill_diagonal(off, 0)
    est_p_inter = off / (node_counts.reshape(-1, 1) @ node_counts.reshape(1, -1) + 1e-6)
    W_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)
    np.fill_diagonal(W, W_intra)
    return float((1 - chi2.cdf(np.abs(W), 1)).mean()) > SBM_WALD_P_THRESHOLD


# ------------------------------------------------------------------------------------------------
# fidelity gate
# ------------------------------------------------------------------------------------------------

def mmd_null_gate(pool, train, refs, B, rng):
    """Per statistic: FAIL only if the original train-vs-ref MMD falls below the 5th percentile of
    a pool null against EVERY reference split.

    The null is MMD(ref, random pool subset of the original train split's size) over B draws.
    Requiring every reference is what keeps the gate from raising false alarms: a genuine
    distribution shift reproduces across references, a chance extreme does not. With six
    comparisons per domain, one floor below a 5th percentile somewhere is unremarkable (the
    released planar pool's spectral floor sits at the 3rd percentile vs test and the 75th vs val).
    """
    from dig2.metrics.spectre_utils import clustering_stats, degree_stats, spectral_stats

    fns = {"degree": degree_stats, "clustering": clustering_stats, "spectral": spectral_stats}
    out = {}
    for name, fn in fns.items():
        per_ref = {}
        for rname, ref in refs.items():
            floor = float(fn(ref, train, is_parallel=False))
            null = np.asarray([float(fn(ref, [pool[i] for i in rng.choice(len(pool), size=len(train),
                                                                         replace=False)], is_parallel=False))
                               for _ in range(B)])
            per_ref[rname] = {"floor_train_vs_ref": floor, "pool_null_median": float(np.median(null)),
                              "pool_null_p05": float(np.quantile(null, 0.05)),
                              "floor_percentile": float((null <= floor).mean()),
                              "below_p05": bool(floor < np.quantile(null, 0.05))}
        out[name] = {"per_ref": per_ref, "pass": not all(v["below_p05"] for v in per_ref.values())}
    return out


def fidelity_gate(domain, pool, original, seed, gate_n=DEFAULT_GATE_N, mmd_null_b=DEFAULT_MMD_NULL_B):
    """Report dict (``report["status"]`` is PASS/FAIL) comparing ``pool`` against ``original``
    ({"train", "val", "test"} -> list of adjacencies). One RNG seeded at ``seed + 1`` draws the
    gate subset and then every MMD null subset, in that order, so the report is reproducible."""
    from scipy.stats import fisher_exact, ks_2samp

    rng = np.random.default_rng(seed + 1)
    gate_idx = rng.choice(len(pool), size=min(gate_n, len(pool)), replace=False)
    pool_nx = to_nx([pool[i] for i in gate_idx])
    tr_nx, va_nx, te_nx = (to_nx(original[s]) for s in ("train", "val", "test"))
    all_orig = to_nx(original["train"] + original["val"] + original["test"])

    report = {"domain": domain, "n_pool": len(pool), "seed": seed, "gate_n": len(pool_nx)}
    n_pool = np.array([g.number_of_nodes() for g in pool_nx])
    n_orig = np.array([g.number_of_nodes() for g in all_orig])
    d_pool = [nx.density(g) for g in pool_nx]
    d_orig = [nx.density(g) for g in all_orig]
    ks_n, ks_d = ks_2samp(n_pool, n_orig), ks_2samp(d_pool, d_orig)
    report["node_count_ks"] = {"stat": float(ks_n.statistic), "p": float(ks_n.pvalue),
                               "pool_range": [int(n_pool.min()), int(n_pool.max())],
                               "spectre_range": [int(n_orig.min()), int(n_orig.max())]}
    report["density_ks"] = {"stat": float(ks_d.statistic), "p": float(ks_d.pvalue),
                            "pool_mean": float(np.mean(d_pool)), "spectre_mean": float(np.mean(d_orig))}
    ok = report["node_count_ks"]["p"] > 0.01 and report["density_ks"]["p"] > 0.01

    if domain == "sbm":
        import graph_tool.all as gt
        k = 0
        for g in pool_nx:
            gt.seed_rng(SBM_CRITERION_GT_SEED)
            k += int(sbm_criterion(g))
        kp, n_ref = SPECTRE_SBM_PASS
        p = float(fisher_exact([[k, len(pool_nx) - k], [kp, n_ref - kp]])[1])
        report["criterion_pass"] = {"pool": [k, len(pool_nx)], "pool_rate": k / len(pool_nx),
                                    "spectre": [kp, n_ref], "spectre_rate": kp / n_ref, "fisher_p": p}
        ok &= p > 0.01

    report["mmd"] = mmd_null_gate(pool_nx, tr_nx, {"val": va_nx, "test": te_nx}, mmd_null_b, rng)
    ok &= all(v["pass"] for v in report["mmd"].values())
    report["status"] = "PASS" if ok else "FAIL"
    return report


# ------------------------------------------------------------------------------------------------
# pool directory
# ------------------------------------------------------------------------------------------------

def ensure_original_raw(domain, source_root):
    """``<source_root>/raw`` of the original dataset, downloaded through the regular loader if it
    is not there yet."""
    raw = os.path.join(source_root, "raw")
    if not all(os.path.exists(os.path.join(raw, f"{s}.pt")) for s in ("train", "val", "test")):
        from dig2.data.spectre import SpectreGraphDataset
        SpectreGraphDataset(dataset_name=domain, split="train", root=source_root)
    return raw


def build_pool(domain, out_root, source_root, n=DEFAULT_N, seed=DEFAULT_SEED, gate_n=DEFAULT_GATE_N,
               mmd_null_b=DEFAULT_MMD_NULL_B, run_gate=True, log=print):
    """Write ``out_root/raw/{train,val,test}.pt`` and, unless ``run_gate`` is False,
    ``out_root/fidelity_gate.json``. Returns the gate report (or None)."""
    if domain not in POOL_DOMAINS:
        raise ValueError(f"domain {domain!r} not in {POOL_DOMAINS}")
    raw = os.path.join(out_root, "raw")
    os.makedirs(raw, exist_ok=True)
    src_raw = ensure_original_raw(domain, source_root)

    t0 = time.time()
    pool = generate(domain, n, seed)
    torch.save(pool, os.path.join(raw, "train.pt"))
    for split in ("val", "test"):
        shutil.copyfile(os.path.join(src_raw, f"{split}.pt"), os.path.join(raw, f"{split}.pt"))
    log(f"[pool] {domain}: generated {len(pool)} graphs ({time.time() - t0:.0f}s) -> {raw}")
    if not run_gate:
        return None

    original = {s: torch.load(os.path.join(src_raw, f"{s}.pt"), weights_only=False)
                for s in ("train", "val", "test")}
    report = fidelity_gate(domain, pool, original, seed, gate_n=gate_n, mmd_null_b=mmd_null_b)
    with open(os.path.join(out_root, "fidelity_gate.json"), "w") as f:
        json.dump(report, f, indent=1)
    log(f"[pool][{domain}][{report['status']}] " + json.dumps(
        {k: report[k] for k in ("node_count_ks", "density_ks", "criterion_pass") if k in report}))
    for name, v in report["mmd"].items():
        pct = "  ".join(f"{r}: pct {x['floor_percentile']:.3f}" for r, x in v["per_ref"].items())
        log(f"[pool][{domain}] MMD {name}: original-train floor percentile in pool null -> {pct} "
            f"-> {'ok' if v['pass'] else 'POOL FARTHER THAN ORIGINAL TRAIN vs every reference'}")
    log(f"[pool] done ({time.time() - t0:.0f}s) -> {os.path.join(out_root, 'fidelity_gate.json')}")
    return report
