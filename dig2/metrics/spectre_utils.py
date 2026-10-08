###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import graph_tool.all as gt

##Navigate to the ./util/orca directory and compile orca.cpp
# g++ -O2 -std=c++11 -o orca orca.cpp
import os
import pickle as pkl
import signal
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import tempfile
import threading
import concurrent.futures
from collections import defaultdict
from collections.abc import Mapping

import pygsp as pg
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from scipy.linalg import eigvalsh
from scipy.stats import chi2
from dig2.metrics.dist_helper import (compute_mmd, gaussian_emd, gaussian, gaussian_tv,)
from torch_geometric.utils import to_networkx

try:
    import wandb
except ImportError:
    wandb = None

from dig2.metrics.abstract import compute_ratios

PRINT_TIME = False
__all__ = [
    "degree_stats",
    "clustering_stats",
    "orbit_stats_all",
    "spectral_stats",
]


# Define a timeout handler
def handler(signum, frame):
    raise TimeoutError


# Set the signal handler for the alarm
signal.signal(signal.SIGALRM, handler)


# ---------------------------------------------------------------------------
# Worker budgeting for metric parallelism.
#
# Several eval runs may share one node (e.g. 8 GPU runs on a 128-core box), so
# the default worker count is affinity-aware and sized for ~8 concurrent jobs.
# Override per-run via cfg `evaluation.metrics_workers` (wired through
# CombinedSpectreSamplingMetrics) or the DIG2_METRICS_WORKERS env var.
# ---------------------------------------------------------------------------

_METRICS_MAX_WORKERS = None


def set_metrics_workers(n):
    global _METRICS_MAX_WORKERS
    _METRICS_MAX_WORKERS = max(1, int(n)) if n else None


def metrics_workers() -> int:
    if _METRICS_MAX_WORKERS is not None:
        return _METRICS_MAX_WORKERS
    env = os.environ.get("DIG2_METRICS_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpus = os.cpu_count() or 1
    return max(1, min(16, n_cpus // 8))


def _process_worker_init():
    """Keep process-pool workers single-threaded so worker_count x thread_count
    cannot oversubscribe the node (numpy/BLAS/graph-tool OpenMP)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        gt.openmp_set_num_threads(1)
    except Exception:
        pass


_ISO_TRAIN_GRAPHS = None


def _iso_train_init(train_graphs):
    _process_worker_init()
    global _ISO_TRAIN_GRAPHS
    _ISO_TRAIN_GRAPHS = train_graphs


def _iso_vs_train_one(fake_g):
    for train_g in _ISO_TRAIN_GRAPHS:
        if nx.faster_could_be_isomorphic(fake_g, train_g):
            if nx.is_isomorphic(fake_g, train_g):
                return True
    return False


def _sbm_validity_one(args):
    G, p_intra, p_inter, strict, refinement_steps = args
    return is_sbm_graph(
        G,
        p_intra=p_intra,
        p_inter=p_inter,
        strict=strict,
        refinement_steps=refinement_steps,
    )


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1 : n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def eigh_worker(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop**2, axis=2)
    hist_range = [0, bound]
    hist = np.array(
        [np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]
    )  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(
    eigvec_ref_list,
    eigval_ref_list,
    eigvec_pred_list,
    eigval_pred_list,
    is_parallel=False,
    compute_emd=False,
):
    """Compute the distance between the eigvector sets.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""

        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_ref_list,
                eigval_ref_list,
                [filters for i in range(len(eigval_ref_list))],
                [bound for i in range(len(eigval_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_pred_list,
                eigval_pred_list,
                [filters for i in range(len(eigval_pred_list))],
                [bound for i in range(len(eigval_pred_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_ref_list[i], eigval_ref_list[i], filters, bound
                )
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_pred_list[i], eigval_pred_list[i], filters, bound
                )
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing spectral filter stats: ", elapsed)
    return mmd_dist


def spectral_stats(
    graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for spectral_density in executor.map(
                spectral_worker,
                graph_pred_list_remove_empty,
                [n_eigvals for i in graph_pred_list_remove_empty],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmd(
            sample_ref,
            sample_pred,
            kernel=gaussian_emd,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10
        )

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts:"


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

_ORCA_TMP_DIR = None
_ORCA_TMP_DIR_LOCK = threading.Lock()


def _orca_tmp_dir():
    global _ORCA_TMP_DIR
    with _ORCA_TMP_DIR_LOCK:
        if _ORCA_TMP_DIR is None:
            _ORCA_TMP_DIR = tempfile.mkdtemp(prefix="orca_")
    return _ORCA_TMP_DIR


ORCA_BIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), "orca", "orca")


def _require_orca_binary():
    """The orbit MMD silently scores every graph as all-zero orbits if ORCA cannot run (see
    `_orca_one`), so a missing binary must fail loudly instead."""
    if not (os.path.isfile(ORCA_BIN) and os.access(ORCA_BIN, os.X_OK)):
        raise FileNotFoundError(
            f"ORCA binary not found at {ORCA_BIN}. Compile it with `bash scripts/build_orca.sh`."
        )


def orca(graph):
    tmp_fname = os.path.join(
        _orca_tmp_dir(),
        f'tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt',
    )
    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()
    output = sp.check_output(
        [
            ORCA_BIN,
            "node",
            "4",
            tmp_fname,
            "std",
        ]
    )

    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(
    graph_ref_list,
    graph_pred_list,
    motif_type="4cycle",
    ground_truth_match=None,
    bins=100,
    compute_emd=False,
):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_ref = np.array(total_counts_ref)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, is_hist=False)
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    return mmd_dist


def _orca_one(G):
    """Compute normalized orbit count vector for one graph (safe for concurrent use)."""
    try:
        orbit_counts = orca(G)
        return np.sum(orbit_counts, axis=0) / max(G.number_of_nodes(), 1)
    except Exception:
        return np.zeros(15)   # ORCA 4-node graphlets → 15 orbit types


def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False, is_parallel=True):
    _require_orca_binary()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            ref_results  = list(executor.map(_orca_one, graph_ref_list))
            pred_results = list(executor.map(_orca_one, graph_pred_list))
    else:
        ref_results  = [_orca_one(G) for G in graph_ref_list]
        pred_results = [_orca_one(G) for G in graph_pred_list]

    total_counts_ref  = np.array(ref_results)
    total_counts_pred = np.array(pred_results)

    if compute_emd:
        return compute_mmd(
            total_counts_ref, total_counts_pred,
            kernel=gaussian, is_hist=False, sigma=30.0,
        )
    return compute_mmd(
        total_counts_ref, total_counts_pred,
        kernel=gaussian_tv, is_hist=False, sigma=30.0,
    )


def eval_acc_tree_graph(G_list, return_validity_by_index=False):
    """``return_validity_by_index``: opt-in (default False -> the historical scalar return,
    byte-for-byte unchanged for every existing caller). When True, also returns the
    per-graph bool list, mirroring eval_acc_sbm_graph. The per-graph verdict was always
    computed here and then discarded; keeping it lets a generated tree's own PASS/FAIL be
    joined against its own per-graph diagnostic trace by list_index, which is what the
    anchor-certificate analysis needs (n = graphs, not n = runs)."""
    validity_by_index = [bool(nx.is_tree(gg)) for gg in G_list]
    # Empty input -> NaN, not ZeroDivisionError. Pre-existing: the original divided by
    # len(G_list) unguarded too. It only surfaces when a run generates no samples (e.g.
    # samples_to_generate=0 for a pure-training timing run), where crashing on an accuracy
    # over zero graphs is strictly worse than reporting "undefined".
    if not G_list:
        return (float("nan"), []) if return_validity_by_index else float("nan")
    acc = float(sum(validity_by_index)) / float(len(G_list))
    if return_validity_by_index:
        return acc, validity_by_index
    return acc


def eval_acc_sbm_graph(
    G_list,
    p_intra=0.3,
    p_inter=0.005,
    strict=True,
    refinement_steps=100,
    is_parallel=False,
    return_per_graph=False,
    validity_by_index=None,
):
    if len(G_list) == 0:
        if return_per_graph:
            return 0.0, []
        return 0.0

    if validity_by_index is None:
        _, validity_by_index = compute_sbm_validity_cache(
            G_list,
            p_intra=p_intra,
            p_inter=p_inter,
            strict=strict,
            refinement_steps=refinement_steps,
        )

    count = float(sum(validity_by_index))
    sbm_acc = count / float(len(G_list))

    if return_per_graph:
        return sbm_acc, validity_by_index
    return sbm_acc


def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))


def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=100):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    A = None
    edge_index = None
    g = None
    state = None
    e = None
    b = None
    node_counts = None
    edge_counts = None
    try:
        A = nx.to_scipy_sparse_array(G, format="coo")
        edge_index = np.column_stack((A.row, A.col))

        g = gt.Graph()
        g.add_vertex(G.number_of_nodes())
        if edge_index.size > 0:
            g.add_edge_list(edge_index)

        state = gt.minimize_blockmodel_dl(g)

        # Refine using merge-split MCMC
        for i in range(refinement_steps):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

        b = state.get_blocks()
        b = gt.contiguous_map(state.get_blocks())
        state = state.copy(b=b)
        e = state.get_matrix()
        n_blocks = state.get_nonempty_B()
        node_counts = state.get_nr().get_array()[:n_blocks]
        edge_counts = e.todense()[:n_blocks, :n_blocks]
        if strict:
            if (
                (node_counts > 40).sum() > 0
                or (node_counts < 20).sum() > 0
                or n_blocks > 5
                or n_blocks < 2
            ):
                return False

        max_intra_edges = node_counts * (node_counts - 1)
        est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

        max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
        np.fill_diagonal(edge_counts, 0)
        est_p_inter = edge_counts / (max_inter_edges + 1e-6)

        W_p_intra = (est_p_intra - p_intra) ** 2 / (
            est_p_intra * (1 - est_p_intra) + 1e-6
        )
        W_p_inter = (est_p_inter - p_inter) ** 2 / (
            est_p_inter * (1 - est_p_inter) + 1e-6
        )

        W = W_p_inter.copy()
        np.fill_diagonal(W, W_p_intra)
        p = 1 - chi2.cdf(abs(W), 1)
        p = p.mean()
        if strict:
            return p > 0.9  # p value < 10 %
        else:
            return p
    except (ValueError, RuntimeError, MemoryError, Exception):
        if strict:
            return False
        else:
            return 0.0
    finally:
        del A, edge_index, g, state, e, b, node_counts, edge_counts


def graph_cache_key(G):
    return (
        G.number_of_nodes(),
        G.number_of_edges(),
        tuple(sorted((min(u, v), max(u, v)) for u, v in G.edges())),
    )


def compute_sbm_validity_cache(
    graph_list,
    p_intra=0.3,
    p_inter=0.005,
    strict=True,
    refinement_steps=100,
):
    # Deduplicate first, then run the expensive graph-tool blockmodel refinement
    # (refinement_steps x minimize per graph) across a bounded process pool.
    # Same per-graph function and cache-key semantics as the serial loop.
    keys = [graph_cache_key(G) for G in graph_list]
    unique_keys = []
    unique_graphs = []
    seen = set()
    for key, G in zip(keys, graph_list):
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
            unique_graphs.append(G)

    n_workers = min(metrics_workers(), max(1, len(unique_graphs)))
    if n_workers > 1 and len(unique_graphs) > 1:
        args = [
            (G, p_intra, p_inter, strict, refinement_steps) for G in unique_graphs
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, initializer=_process_worker_init
        ) as executor:
            results = list(executor.map(_sbm_validity_one, args))
    else:
        results = [
            is_sbm_graph(
                G,
                p_intra=p_intra,
                p_inter=p_inter,
                strict=strict,
                refinement_steps=refinement_steps,
            )
            for G in unique_graphs
        ]

    validity_by_key = dict(zip(unique_keys, results))
    validity_by_index = [validity_by_key[key] for key in keys]
    return validity_by_key, validity_by_index


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    # Each fake graph is checked against the train set independently, so the
    # outer loop parallelizes across a bounded process pool. The train list is
    # shipped to each worker once via the pool initializer, not per task.
    n_workers = min(metrics_workers(), max(1, len(fake_graphs)))
    if n_workers > 1 and len(fake_graphs) > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_iso_train_init,
            initargs=(train_graphs,),
        ) as executor:
            results = list(executor.map(_iso_vs_train_one, fake_graphs))
        count = sum(bool(r) for r in results)
    else:
        count = 0
        for fake_g in fake_graphs:
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count += 1
                        break
    return count / float(len(fake_graphs))


def eval_fraction_unique_non_isomorphic_valid(
    fake_graphs,
    train_graphs,
    validity_func=(lambda x: True),
    validity_by_index=None,
    validity_by_key=None,
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for i, fake_g in enumerate(fake_graphs):
        unique = True

        for fake_old in fake_evaluated:
            try:
                # Set the alarm for 60 seconds
                signal.alarm(60)
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
            except TimeoutError:
                print("Timeout: Skipping this iteration")
                continue
            finally:
                # Disable the alarm
                signal.alarm(0)
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_by_index is not None:
                    is_valid = validity_by_index[i]
                elif validity_by_key is not None:
                    is_valid = validity_by_key[graph_cache_key(fake_g)]
                else:
                    is_valid = validity_func(fake_g)
                if is_valid:
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
        float(len(fake_graphs)) - count_non_unique - count_isomorphic
    ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


class SpectreSamplingMetrics(nn.Module):
    def __init__(self, datamodule, compute_emd, metrics_list, model=None):
        super().__init__()

        self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())
        self.num_graphs_test = len(self.test_graphs)
        self.num_graphs_val = len(self.val_graphs)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list
        self.model = model

        # Store for wavelet computaiton
        self.val_ref_eigvals, self.val_ref_eigvecs = compute_list_eigh(self.val_graphs)
        self.test_ref_eigvals, self.test_ref_eigvecs = compute_list_eigh(
            self.test_graphs
        )

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                networkx_graphs.append(
                    to_networkx(
                        data,
                        node_attrs=None,
                        edge_attrs=None,
                        to_undirected=True,
                        remove_self_loops=True,
                    )
                )
        return networkx_graphs

    def forward(
        self,
        generated_graphs: list,
        ref_metrics,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
        labels=None,
    ):
        reference_graphs = self.test_graphs if test else self.val_graphs
        if local_rank == 0:
            print(
                f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(reference_graphs)}"
                f" test graphs -- emd computation: {self.compute_emd}"
            )
        networkx_graphs = []
        adjacency_matrices = []
        if local_rank == 0:
            print("Building networkx graphs...")
        for graph in generated_graphs:
            node_types, edge_types = graph
            A = edge_types.bool().cpu().numpy()
            if A.shape[0] != A.shape[1]:
                min_dim = min(A.shape[0], A.shape[1])
                A = A[:min_dim, :min_dim]
                A = np.maximum(A, A.T)
            adjacency_matrices.append(A)

            nx_graph = nx.from_numpy_array(A)
            nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
            networkx_graphs.append(nx_graph)

        to_log = {}
        np.savez("generated_adjs.npz", *adjacency_matrices)

        if "degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing degree stats..")
            degree = degree_stats(
                reference_graphs,
                networkx_graphs,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            if wandb is not None and wandb.run:
                wandb.run.summary["degree"] = degree
            to_log["degree"] = degree

        # val_eigvals = [graph["eigval"][1:self.k + 1].cpu().detach().numpy() for graph in self.val]
        # train_eigvals = [graph["eigval"][1:self.k + 1].cpu().detach().numpy() for graph in self.train]

        # eigval_stats(eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False)
        # spectral_filter_stats(eigvec_ref_list, eigval_ref_list, eigvec_pred_list, eigval_pred_list, is_parallel=False,
        #                       compute_emd=False)          # This is the one called wavelet

        if "wavelet" in self.metrics_list:
            if local_rank == 0:
                print("Computing wavelet stats...")

            ref_eigvecs = self.test_ref_eigvecs if test else self.val_ref_eigvecs
            ref_eigvals = self.test_ref_eigvals if test else self.val_ref_eigvals

            pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(networkx_graphs)
            wavelet = spectral_filter_stats(
                eigvec_ref_list=ref_eigvecs,
                eigval_ref_list=ref_eigvals,
                eigvec_pred_list=pred_graph_eigvecs,
                eigval_pred_list=pred_graph_eigvals,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
            to_log["wavelet"] = wavelet
            if wandb is not None and wandb.run:
                wandb.run.summary["wavelet"] = wavelet

        if "spectre" in self.metrics_list:
            if local_rank == 0:
                print("Computing spectre stats...")
            spectre = spectral_stats(
                reference_graphs,
                networkx_graphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )

            to_log["spectre"] = spectre
            if wandb is not None and wandb.run:
                wandb.run.summary["spectre"] = spectre

        if "clustering" in self.metrics_list:
            if local_rank == 0:
                print("Computing clustering stats...")
            clustering = clustering_stats(
                reference_graphs,
                networkx_graphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log["clustering"] = clustering
            if wandb is not None and wandb.run:
                wandb.run.summary["clustering"] = clustering

        if "motif" in self.metrics_list:
            if local_rank == 0:
                print("Computing motif stats")
            motif = motif_stats(
                reference_graphs,
                networkx_graphs,
                motif_type="4cycle",
                ground_truth_match=None,
                bins=100,
                compute_emd=self.compute_emd,
            )
            to_log["motif"] = motif
            if wandb is not None and wandb.run:
                wandb.run.summary["motif"] = motif

        if "orbit" in self.metrics_list:
            if local_rank == 0:
                print("Computing orbit stats...")
            orbit = orbit_stats_all(
                reference_graphs, networkx_graphs, compute_emd=self.compute_emd
            )
            to_log["orbit"] = orbit
            if wandb is not None and wandb.run:
                wandb.run.summary["orbit"] = orbit

        if "sbm" in self.metrics_list:
            if local_rank == 0:
                print("Computing accuracy...")
            sbm_acc, sbm_validity_by_index = eval_acc_sbm_graph(
                networkx_graphs,
                refinement_steps=100,
                strict=True,
                return_per_graph=True,
            )
            to_log["sbm_acc"] = sbm_acc
            if wandb is not None and wandb.run:
                wandb.run.summary["sbm_acc"] = sbm_acc
        else:
            sbm_validity_by_index = None

        if "planar" in self.metrics_list:
            if local_rank == 0:
                print("Computing planar accuracy...")
            planar_acc = eval_acc_planar_graph(networkx_graphs)
            to_log["planar_acc"] = planar_acc
            if wandb is not None and wandb.run:
                wandb.run.summary["planar_acc"] = planar_acc

        if "tree" in self.metrics_list:
            if local_rank == 0:
                print("Computing tree accuracy...")
            tree_acc = eval_acc_tree_graph(networkx_graphs)
            to_log["tree_acc"] = tree_acc
            if wandb is not None and wandb.run:
                wandb.run.summary["tree_acc"] = tree_acc

        if (
            "sbm" in self.metrics_list
            or "planar" in self.metrics_list
            or "tree" in self.metrics_list
        ):
            if local_rank == 0:
                print("Computing all fractions...")
            if "sbm" in self.metrics_list:
                validity_func = is_sbm_graph
            elif "planar" in self.metrics_list:
                validity_func = is_planar_graph
            elif "tree" in self.metrics_list:
                validity_func = nx.is_tree
            else:
                validity_func = None
            (
                frac_unique,
                frac_unique_non_isomorphic,
                fraction_unique_non_isomorphic_valid,
            ) = eval_fraction_unique_non_isomorphic_valid(
                networkx_graphs,
                self.train_graphs,
                validity_func,
                validity_by_index=sbm_validity_by_index if "sbm" in self.metrics_list else None,
            )
            frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(
                networkx_graphs, self.train_graphs
            )
            to_log.update(
                {
                    "sampling/frac_unique": frac_unique,
                    "sampling/frac_unique_non_iso": frac_unique_non_isomorphic,
                    "sampling/frac_unic_non_iso_valid": fraction_unique_non_isomorphic_valid,
                    "sampling/frac_non_iso": frac_non_isomorphic,
                }
            )

        ratios = compute_ratios(
            gen_metrics=to_log,
            ref_metrics=ref_metrics["test"] if test else ref_metrics["val"],
            metrics_keys=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )
        to_log.update(ratios)

        if local_rank == 0:
            print("Sampling statistics", to_log)
        if wandb is not None and wandb.run:
            wandb.log(to_log, commit=False)

        return to_log

    def reset(self):
        pass


class CombinedSpectreSamplingMetrics(nn.Module):
    """Per-domain evaluation metrics for combined SPECTRE datasets.

    This class evaluates the same generated set against each reference domain
    separately and logs prefixed metrics, e.g.:
      - domain/tree/degree
      - domain/planar/planar_acc
      - domain_macro/degree
    """

    SHARED_METRICS = ["degree", "clustering", "orbit", "spectre", "wavelet"]
    SYNTHETIC_DOMAIN_NAMES = {"sbm", "comm20", "planar", "tree"}
    MOLECULAR_DOMAIN_NAMES = {"qm9", "zinc", "moses", "guacamol"}


    def __init__(
        self,
        datamodule,
        compute_emd=False,
        shared_metrics=None,
        domain_validity=None,
        domain_names=None,
        max_loader_batches=None,
    ):
        super().__init__()
        import time
        self.compute_emd = compute_emd
        self.shared_metrics = shared_metrics or self.SHARED_METRICS
        self.domain_validity = domain_validity or {
            "tree":   "tree",
            "planar": "planar",
            "sbm":    "sbm",
            "comm20": None,
        }
        self.domain_names = domain_names or list(getattr(datamodule, "dataset_names", []))
        eval_cfg = getattr(getattr(datamodule, "cfg", None), "evaluation", None)
        _workers_cfg = getattr(eval_cfg, "metrics_workers", None)
        if _workers_cfg is not None:
            set_metrics_workers(_workers_cfg)
        mixed_cfg = getattr(eval_cfg, "mixed", None)
        self.startup_verbose = bool(getattr(mixed_cfg, "startup_verbose", True))
        self.startup_log_every_batches = max(1, int(getattr(mixed_cfg, "startup_log_every_batches", 1)))
        self.startup_log_memory = bool(getattr(mixed_cfg, "startup_log_memory", True))
        self._cache_nx_graphs = bool(getattr(mixed_cfg, "cache_nx_graphs", False))
        def _log(msg):
            if self.startup_verbose:
                print(f"[spectre-startup] {msg}", flush=True)

        def _log_mem(prefix):
            if not self.startup_log_memory:
                return
            try:
                import resource
                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_gb = float(rss_kb) / (1024.0 * 1024.0)
                print(f"[spectre-startup][mem] {prefix}: rss_max_gb={rss_gb:.3f}", flush=True)
            except Exception:
                pass

        self._startup_log = _log
        self._startup_log_mem = _log_mem

        self.max_loader_batches = max_loader_batches
        self.synthetic_domains = [
            str(name).strip().lower()
            for name in self.domain_names
            if str(name).strip().lower() in self.SYNTHETIC_DOMAIN_NAMES
        ]

        # Try to build synthetic-only loaders so we skip iterating through molecular
        # samples that precede planar/tree in the sequential val/test manifest.
        _synth_train = self._build_synthetic_loader_if_available(datamodule, "train")
        _synth_val   = self._build_synthetic_loader_if_available(datamodule, "val")
        _synth_test  = self._build_synthetic_loader_if_available(datamodule, "test")
        _using_synth = _synth_train is not None and _synth_val is not None and _synth_test is not None

        if _using_synth:
            # No batch cap needed — synthetic-only loaders contain only the domains we need.
            self.max_loader_batches = None
        elif self.max_loader_batches is None:
            lower_names = [str(name).strip().lower() for name in self.domain_names]
            if any(name in self.MOLECULAR_DOMAIN_NAMES for name in lower_names):
                # Fallback guard: prevent extremely long startup when iterating the full
                # sequential loader that contains large molecular splits.
                self.max_loader_batches = 8

        train_ref_loader = _synth_train if _using_synth else datamodule.train_dataloader()
        val_ref_loader   = _synth_val   if _using_synth else datamodule.val_dataloader()
        test_ref_loader  = _synth_test  if _using_synth else datamodule.test_dataloader()

        ctor_t0 = time.perf_counter()
        self._startup_log(
            f"init domain_names={self.domain_names}, synthetic_domains={self.synthetic_domains}, "
            f"max_loader_batches={self.max_loader_batches}, "
            f"using_synthetic_only_loader={_using_synth}"
        )

        t = time.perf_counter()
        self._startup_log("building train_graphs_by_domain")
        self.train_graphs_by_domain = self._load_or_build_nx_by_domain(
            train_ref_loader,
            split_name="train",
            datamodule=datamodule,
        )
        self._startup_log(f"train_graphs_by_domain done elapsed_s={time.perf_counter()-t:.2f}")
        self._startup_log_mem("after_train_loader_to_nx")

        t = time.perf_counter()
        self._startup_log("building val_graphs_by_domain")
        self.val_graphs_by_domain = self._load_or_build_nx_by_domain(
            val_ref_loader,
            split_name="val",
            datamodule=datamodule,
        )
        self._startup_log(f"val_graphs_by_domain done elapsed_s={time.perf_counter()-t:.2f}")
        self._startup_log_mem("after_val_loader_to_nx")

        t = time.perf_counter()
        self._startup_log("building test_graphs_by_domain")
        self.test_graphs_by_domain = self._load_or_build_nx_by_domain(
            test_ref_loader,
            split_name="test",
            datamodule=datamodule,
        )
        self._startup_log(f"test_graphs_by_domain done elapsed_s={time.perf_counter()-t:.2f}")
        self._startup_log_mem("after_test_loader_to_nx")

        t = time.perf_counter()
        self._startup_log("computing val eigen references")
        self.val_eigs_by_domain = {
            domain: compute_list_eigh(graphs)
            for domain, graphs in self.val_graphs_by_domain.items()
        }
        self._startup_log(f"val eigen references done elapsed_s={time.perf_counter()-t:.2f}")
        self._startup_log_mem("after_val_eigs")

        t = time.perf_counter()
        self._startup_log("computing test eigen references")
        self.test_eigs_by_domain = {
            domain: compute_list_eigh(graphs)
            for domain, graphs in self.test_graphs_by_domain.items()
        }
        self._startup_log(f"test eigen references done elapsed_s={time.perf_counter()-t:.2f}")
        self._startup_log_mem("after_test_eigs")

        t = time.perf_counter()
        self._startup_log("initializing per-domain reference metric caches")
        self.ref_metrics_by_domain = self._init_domain_local_refs(datamodule)
        self._startup_log(f"domain reference metric init done elapsed_s={time.perf_counter()-t:.2f}")
        self._startup_log_mem("after_domain_ref_init")
        self._startup_log(f"CombinedSpectreSamplingMetrics init done total_elapsed_s={time.perf_counter()-ctor_t0:.2f}")

    def _domain_ref_cache_path(self, domain, datamodule):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        ds_cfg = getattr(getattr(datamodule, "cfg", None), "dataset", None)
        data_root = os.path.join(base_path, str(getattr(ds_cfg, "datadir", "data")))
        domain_root = os.path.join(data_root, str(domain).strip().lower())
        os.makedirs(domain_root, exist_ok=True)
        return os.path.join(domain_root, "ref_metrics.pkl")

    def _nx_graphs_cache_path(self, split: str, datamodule) -> str:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cfg = getattr(datamodule, "cfg", None)
        ds_cfg = getattr(cfg, "dataset", None)
        unified_cfg = getattr(ds_cfg, "unified", None)
        datadir = str(getattr(ds_cfg, "datadir", "data")) if ds_cfg else "data"
        sv = int(getattr(unified_cfg, "schema_version", 1)) if unified_cfg else 1
        cache_dir = os.path.join(base_path, datadir, "unified", "nx_cache")
        os.makedirs(cache_dir, exist_ok=True)
        mlb = "all" if self.max_loader_batches is None else str(int(self.max_loader_batches))
        domain_tag = "-".join(sorted(str(d).strip().lower() for d in self.synthetic_domains))
        return os.path.join(cache_dir, f"{split}_nx_v{sv}_mlb{mlb}_dom{domain_tag[:48]}.pkl")

    def _load_or_build_nx_by_domain(self, loader, split_name: str, datamodule):
        import pickle, time as _time
        if not self._cache_nx_graphs:
            return self._loader_to_nx_by_domain(loader, split_name=split_name)

        cache_path = self._nx_graphs_cache_path(split_name, datamodule)

        if os.path.exists(cache_path):
            try:
                t = _time.perf_counter()
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict) and "by_domain" in payload:
                    by_domain = payload["by_domain"]
                    saved_counts = payload.get("graph_counts", {})
                    # Validate counts are consistent with what was saved.
                    if (
                        len(by_domain) > 0
                        and all(isinstance(v, list) for v in by_domain.values())
                        and all(len(by_domain[d]) == saved_counts.get(d, -1) for d in by_domain)
                    ):
                        self._startup_log(
                            f"nx_cache_hit split={split_name} path={cache_path} "
                            f"domains={list(by_domain.keys())} elapsed_s={_time.perf_counter()-t:.2f}"
                        )
                        return by_domain
            except Exception as e:
                self._startup_log(f"nx_cache_load_failed split={split_name} err={e}")

        by_domain = self._loader_to_nx_by_domain(loader, split_name=split_name)

        try:
            t = _time.perf_counter()
            graph_counts = {d: len(graphs) for d, graphs in by_domain.items()}
            with open(cache_path, "wb") as f:
                pickle.dump({"by_domain": by_domain, "graph_counts": graph_counts, "split": split_name}, f)
            self._startup_log(
                f"nx_cache_saved split={split_name} path={cache_path} "
                f"counts={graph_counts} elapsed_s={_time.perf_counter()-t:.2f}"
            )
        except Exception as e:
            self._startup_log(f"nx_cache_save_failed split={split_name} err={e}")

        return by_domain

    def _compute_domain_reference_metrics(self, domain):
        train_graphs = list(self.train_graphs_by_domain.get(domain, []))
        val_graphs = list(self.val_graphs_by_domain.get(domain, []))
        test_graphs = list(self.test_graphs_by_domain.get(domain, []))

        val_ref = self._compute_domain_metrics(
            reference_graphs=val_graphs,
            generated_graphs=train_graphs,
            domain=domain,
            test=False,
        )
        test_ref = self._compute_domain_metrics(
            reference_graphs=test_graphs,
            generated_graphs=train_graphs,
            domain=domain,
            test=True,
        )
        return {"val": val_ref, "test": test_ref}

    def _init_domain_local_refs(self, datamodule):
        import time
        refs = {}
        for domain in self.synthetic_domains:
            domain = str(domain).strip().lower()
            if domain not in self.val_graphs_by_domain:
                continue
            cache_path = self._domain_ref_cache_path(domain, datamodule)
            domain_refs = None
            t = time.perf_counter()
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        maybe_refs = pkl.load(f)
                    if isinstance(maybe_refs, Mapping):
                        domain_refs = maybe_refs
                        self._startup_log(
                            f"domain={domain} cache_hit path={cache_path} elapsed_s={time.perf_counter()-t:.2f}"
                        )
                except Exception:
                    domain_refs = None

            if domain_refs is None or "val" not in domain_refs or "test" not in domain_refs:
                self._startup_log(f"domain={domain} cache_miss -> computing reference metrics")
                t_compute = time.perf_counter()
                domain_refs = self._compute_domain_reference_metrics(domain)
                self._startup_log(
                    f"domain={domain} reference metric compute done elapsed_s={time.perf_counter()-t_compute:.2f}"
                )
                t_write = time.perf_counter()
                with open(cache_path, "wb") as f:
                    pkl.dump(domain_refs, f)
                self._startup_log(
                    f"domain={domain} cache_write_done path={cache_path} elapsed_s={time.perf_counter()-t_write:.2f}"
                )

            refs[domain] = domain_refs
        return refs

    def _build_synthetic_loader_if_available(self, datamodule, split: str):
        """
        Build a DataLoader containing only synthetic domain samples for the given split.

        Uses dataset.domain_indices (available on UnifiedDataset) to create a
        Subset that skips molecular domains entirely. This avoids iterating through tens
        of thousands of zinc/qm9 samples to reach planar/tree in the sequential manifest.

        Returns the loader on success, or None to fall back to the full dataloader.
        """
        try:
            from torch.utils.data import Subset
            from torch_geometric.loader import DataLoader as PyGDataLoader

            dataset = getattr(datamodule, f"{split}_dataset", None)
            if dataset is None:
                return None
            domain_indices_map = getattr(dataset, "domain_indices", None)
            if not isinstance(domain_indices_map, dict) or not domain_indices_map:
                return None

            synth_indices = []
            for domain in self.synthetic_domains:
                idxs = domain_indices_map.get(str(domain).strip().lower(), [])
                synth_indices.extend(idxs)
            if not synth_indices:
                return None

            synth_indices = sorted(synth_indices)
            subset = Subset(dataset, synth_indices)
            cfg = getattr(datamodule, "cfg", None)
            bs = max(32, int(getattr(getattr(cfg, "train", None), "batch_size", 32) or 32))
            self._startup_log(
                f"synthetic_loader split={split} n_indices={len(synth_indices)} batch_size={bs}"
            )
            return PyGDataLoader(subset, batch_size=bs, shuffle=False, num_workers=0)
        except Exception as exc:
            self._startup_log(f"synthetic_loader_unavailable split={split} err={exc}")
            return None

    def _loader_to_nx_by_domain(self, loader, split_name="unknown"):
        import time
        by_domain = defaultdict(list)
        t0 = time.perf_counter()
        for batch_idx, batch in enumerate(loader):
            if self.max_loader_batches is not None and batch_idx >= int(self.max_loader_batches):
                self._startup_log(
                    f"split={split_name} reached max_loader_batches={self.max_loader_batches} at batch_idx={batch_idx}"
                )
                break
            for data in batch.to_data_list():
                domain = getattr(data, "dataset_name", "unknown")
                by_domain[str(domain)].append(
                    to_networkx(
                        data,
                        node_attrs=None,
                        edge_attrs=None,
                        to_undirected=True,
                        remove_self_loops=True,
                    )
                )
            if self.startup_verbose and ((batch_idx + 1) % self.startup_log_every_batches == 0):
                domain_counts = {k: len(v) for k, v in by_domain.items()}
                self._startup_log(
                    f"split={split_name} batch_idx={batch_idx} elapsed_s={time.perf_counter()-t0:.2f} "
                    f"domain_graph_counts={domain_counts}"
                )
                self._startup_log_mem(f"split={split_name} batch_idx={batch_idx}")
        return dict(by_domain)

    def _build_generated_nx(self, generated_graphs):
        def _convert_one(graph):
            if isinstance(graph, Mapping) and "graph" in graph:
                graph = graph["graph"]
            node_types, edge_types = graph
            A = edge_types.bool().cpu().numpy()
            if A.shape[0] != A.shape[1]:
                min_dim = min(A.shape[0], A.shape[1])
                A = A[:min_dim, :min_dim]
                A = np.maximum(A, A.T)
            nx_graph = nx.from_numpy_array(A)
            nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
            return nx_graph, A

        with concurrent.futures.ThreadPoolExecutor(max_workers=metrics_workers()) as executor:
            results = list(executor.map(_convert_one, generated_graphs))
        if not results:
            return [], []
        networkx_graphs, adjacency_matrices = zip(*results)
        return list(networkx_graphs), list(adjacency_matrices)

    def _compute_domain_metrics(self, reference_graphs, generated_graphs, domain, test=False):
        domain_log = {}
        domain = str(domain).strip().lower()
        if domain not in self.synthetic_domains:
            return domain_log
        if not reference_graphs:
            return domain_log

        if "degree" in self.shared_metrics:
            domain_log["degree"] = degree_stats(
                reference_graphs,
                generated_graphs,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
        if "wavelet" in self.shared_metrics:
            ref_eigvals, ref_eigvecs = (
                self.test_eigs_by_domain[domain]
                if test
                else self.val_eigs_by_domain[domain]
            )
            pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(generated_graphs)
            domain_log["wavelet"] = spectral_filter_stats(
                eigvec_ref_list=ref_eigvecs,
                eigval_ref_list=ref_eigvals,
                eigvec_pred_list=pred_graph_eigvecs,
                eigval_pred_list=pred_graph_eigvals,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
        if "spectre" in self.shared_metrics:
            domain_log["spectre"] = spectral_stats(
                reference_graphs,
                generated_graphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )
        if "clustering" in self.shared_metrics:
            domain_log["clustering"] = clustering_stats(
                reference_graphs,
                generated_graphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
        if "orbit" in self.shared_metrics:
            domain_log["orbit"] = orbit_stats_all(
                reference_graphs, generated_graphs, compute_emd=self.compute_emd
            )

        validity_tag = self.domain_validity.get(domain)
        if validity_tag == "tree":
            domain_log["tree_acc"] = eval_acc_tree_graph(generated_graphs)
            validity_func = nx.is_tree
        elif validity_tag == "planar":
            domain_log["planar_acc"] = eval_acc_planar_graph(generated_graphs)
            validity_func = is_planar_graph
        elif validity_tag == "sbm":
            domain_log["sbm_acc"], sbm_validity_by_index = eval_acc_sbm_graph(
                generated_graphs,
                refinement_steps=100,
                strict=True,
                return_per_graph=True,
            )
            validity_func = is_sbm_graph
        else:
            validity_func = None
            sbm_validity_by_index = None

        if validity_func is not None and domain in self.train_graphs_by_domain:
            (
                frac_unique,
                frac_unique_non_isomorphic,
                fraction_unique_non_isomorphic_valid,
            ) = eval_fraction_unique_non_isomorphic_valid(
                generated_graphs,
                self.train_graphs_by_domain[domain],
                validity_func,
                validity_by_index=sbm_validity_by_index if validity_tag == "sbm" else None,
            )
            frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(
                generated_graphs, self.train_graphs_by_domain[domain]
            )
            domain_log.update(
                {
                    "sampling/frac_unique": frac_unique,
                    "sampling/frac_unique_non_iso": frac_unique_non_isomorphic,
                    "sampling/frac_unic_non_iso_valid": fraction_unique_non_isomorphic_valid,
                    "sampling/frac_non_iso": frac_non_isomorphic,
                }
            )

        return domain_log

    def forward(
        self,
        generated_graphs: list,
        ref_metrics,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
        labels=None,
    ):
        generated_nx, adjacency_matrices = self._build_generated_nx(generated_graphs)
        np.savez("generated_adjs.npz", *adjacency_matrices)

        reference_all_domains = self.test_graphs_by_domain if test else self.val_graphs_by_domain
        reference_by_domain = {
            str(domain).strip().lower(): graphs
            for domain, graphs in reference_all_domains.items()
            if str(domain).strip().lower() in self.synthetic_domains
        }
        to_log = {}

        # Optional conditioned evaluation path: split generated graphs by provided one-hot labels.
        generated_by_domain = None
        # Preferred path for mixed sampling: explicit dataset_name metadata per sample.
        if len(generated_graphs) > 0 and isinstance(generated_graphs[0], Mapping):
            generated_by_domain = defaultdict(list)
            for graph_meta, graph_nx in zip(generated_graphs, generated_nx):
                domain_name = str(graph_meta.get("dataset_name", "")).strip().lower()
                if domain_name:
                    generated_by_domain[domain_name].append(graph_nx)
            generated_by_domain = dict(generated_by_domain)

        if labels is not None and len(labels) == len(generated_nx) and len(self.domain_names) > 0:
            generated_by_domain = defaultdict(list)
            for graph, label in zip(generated_nx, labels):
                if label is None:
                    continue
                if isinstance(label, torch.Tensor) and label.numel() > 0:
                    domain_idx = int(torch.argmax(label).item())
                    if 0 <= domain_idx < len(self.domain_names):
                        generated_by_domain[self.domain_names[domain_idx]].append(graph)
            generated_by_domain = dict(generated_by_domain)

        for domain, reference_graphs in reference_by_domain.items():
            eval_graphs = generated_nx
            log_prefix = f"domain/{domain}"
            if generated_by_domain is not None:
                eval_graphs = generated_by_domain.get(domain, [])
                log_prefix = f"cond/domain/{domain}"

            if len(eval_graphs) == 0:
                continue

            if local_rank == 0:
                print(
                    f"Computing domain metrics for {domain}: {len(eval_graphs)} generated vs {len(reference_graphs)} reference"
                )
            domain_metrics = self._compute_domain_metrics(
                reference_graphs=reference_graphs,
                generated_graphs=eval_graphs,
                domain=domain,
                test=test,
            )

            ratio_base_metrics = [m for m in self.shared_metrics if m in domain_metrics]
            if len(ratio_base_metrics) > 0:
                split_key = "test" if test else "val"
                domain_ref_metrics = (
                    self.ref_metrics_by_domain.get(str(domain).strip().lower(), {})
                    .get(split_key)
                )
                if domain_ref_metrics is None and local_rank == 0:
                    print(
                        f"WARNING: Missing domain-local reference metrics for domain '{domain}' ({split_key}). Skipping ratio metrics."
                    )
                ratios = compute_ratios(
                    gen_metrics=domain_metrics,
                    ref_metrics=domain_ref_metrics,
                    metrics_keys=ratio_base_metrics,
                )
                domain_metrics.update(ratios)

            for key, value in domain_metrics.items():
                to_log[f"{log_prefix}/{key}"] = value

        macro_metric_names = list(self.shared_metrics)
        macro_metric_names.extend([f"{metric_name}_ratio" for metric_name in self.shared_metrics])
        macro_metric_names.append("average_ratio")

        # Macro-average metrics that exist across evaluated domains.
        for metric_name in macro_metric_names:
            values = []
            for domain in reference_by_domain.keys():
                metric_key = f"domain/{domain}/{metric_name}"
                if generated_by_domain is not None:
                    metric_key = f"cond/domain/{domain}/{metric_name}"
                if metric_key in to_log:
                    values.append(float(to_log[metric_key]))
            if values:
                macro_prefix = "domain_macro"
                if generated_by_domain is not None:
                    macro_prefix = "cond/domain_macro"
                to_log[f"{macro_prefix}/{metric_name}"] = float(np.mean(values))

        if local_rank == 0:
            print("Combined sampling statistics", to_log)
        if wandb is not None and wandb.run:
            wandb.log(to_log, commit=False)

        return to_log

    def reset(self):
        pass





