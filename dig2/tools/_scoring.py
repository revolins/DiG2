"""Per-graph verdicts used by `dig2.tools.lit_protocol`.

The set-level metrics in `dig2.metrics` report one rate per domain; the literature protocol
needs a verdict per graph (to form blocks, Wilson intervals and per-graph disagreements between
instruments). Everything here reuses `dig2.metrics.rdkit_functions` / the unified semantic
vocabulary, so a per-graph verdict is the same predicate the run-level metric counts.
"""
import os
import pickle

import networkx as nx
import numpy as np
import torch

# graph-tool's block-model fit is stochastic: every structural verdict is preceded by
# gt.seed_rng(GT_SEED) so that rescoring the same graphs gives the same answer.
GT_SEED = 12345

SBM_P_INTRA = 0.3
SBM_P_INTER = 0.005
SBM_BLOCK_SIZE_RANGE = (20, 40)
SBM_BLOCK_COUNT_RANGE = (2, 5)
SBM_WALD_P_THRESHOLD = 0.9
SBM_REFINEMENT_STEPS = 100


def wilson(k, n, z=1.96):
    """(rate, lo, hi) Wilson score interval; correct near 0 and 1, where the normal
    approximation is not (structural domains are scored on a few hundred graphs)."""
    import math
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------------------------------- structural

def _fit_sbm_blockmodel(G, refinement_steps=SBM_REFINEMENT_STEPS):
    """The same graph-tool fit as `spectre_utils.is_sbm_graph` (minimize_blockmodel_dl, then
    merge-split MCMC refinement), returning block sizes and estimated block densities."""
    import graph_tool.all as gt

    A = nx.to_scipy_sparse_array(G, format="coo")
    edge_index = np.column_stack((A.row, A.col))
    g = gt.Graph()
    g.add_vertex(G.number_of_nodes())
    if edge_index.size > 0:
        g.add_edge_list(edge_index)

    state = gt.minimize_blockmodel_dl(g)
    for _ in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = np.asarray(e.todense())[:n_blocks, :n_blocks]

    max_intra = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra + 1e-6)
    max_inter = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    ec_off = edge_counts.copy()
    np.fill_diagonal(ec_off, 0)
    est_p_inter = ec_off / (max_inter + 1e-6)
    return {"n_blocks": int(n_blocks), "block_sizes": node_counts.astype(int).tolist(),
            "est_p_intra": est_p_intra, "est_p_inter": est_p_inter}


def _wald_p(fit, p_intra=SBM_P_INTRA, p_inter=SBM_P_INTER):
    """Mean Wald p-value of the fitted block densities against the generator's (p_intra,
    p_inter), computed ELEMENTWISE. `spectre_utils.is_sbm_graph` builds the same statistic on
    np.matrix, where `**` is a matrix power, so its p-value differs; the strict instrument uses
    the elementwise form and the literature variants keep is_sbm_graph unchanged."""
    from scipy.stats import chi2
    n_blocks = int(fit["n_blocks"])
    est_p_intra = np.asarray(fit["est_p_intra"], dtype=float)
    est_p_inter = np.asarray(fit["est_p_inter"], dtype=float).reshape(n_blocks, n_blocks)
    W_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)
    W = W_inter.copy()
    np.fill_diagonal(W, W_intra)
    p = 1 - chi2.cdf(np.abs(W), 1)
    return float(p.mean())


def strict_structural_verdict(G, domain):
    """{"ok", "fail_reason", "n_blocks", "block_sizes", "wald_p"} for one graph.

    planar / tree: `spectre_utils.is_planar_graph` / `nx.is_tree`. sbm: one block-model fit; the
    graph passes with 2-5 blocks of 20-40 nodes and an elementwise mean Wald p-value above 0.9.
    The caller seeds graph-tool first."""
    out = {"n_blocks": None, "block_sizes": None, "wald_p": None}
    if domain in ("planar", "tree"):
        try:
            if domain == "planar":
                from dig2.metrics.spectre_utils import is_planar_graph
                ok = is_planar_graph(G)
            else:
                ok = nx.is_tree(G)
        except Exception:  # noqa: BLE001  (nx.is_tree raises on the empty graph)
            return {**out, "ok": False, "fail_reason": "fit_error"}
        return {**out, "ok": bool(ok), "fail_reason": "pass" if ok else f"not_{domain}"}

    try:
        fit = _fit_sbm_blockmodel(G)
    except Exception:  # noqa: BLE001  (graph-tool raises on degenerate graphs; is_sbm_graph catches too)
        return {**out, "ok": False, "fail_reason": "fit_error"}
    out.update(n_blocks=fit["n_blocks"], block_sizes=fit["block_sizes"])
    sizes = np.array(fit["block_sizes"])
    if not SBM_BLOCK_COUNT_RANGE[0] <= fit["n_blocks"] <= SBM_BLOCK_COUNT_RANGE[1]:
        return {**out, "ok": False, "fail_reason": "block_count_out_of_range"}
    if (sizes < SBM_BLOCK_SIZE_RANGE[0]).any() or (sizes > SBM_BLOCK_SIZE_RANGE[1]).any():
        return {**out, "ok": False, "fail_reason": "block_size_out_of_range"}
    wald_p = _wald_p(fit)
    ok = wald_p > SBM_WALD_P_THRESHOLD
    return {**out, "wald_p": wald_p, "ok": bool(ok), "fail_reason": "pass" if ok else "density_mismatch"}


# --------------------------------------------------------------------------------- molecular

def semantic_remaps_for_domain(domain, remove_h=False):
    """(global->local atom remap, global->local edge remap, local atom_decoder) for one molecular
    domain of the unified semantic vocabulary -- the maps the evaluator applies to generated
    graphs. qm9's node map depends on remove_h (C,N,O,F vs H,C,N,O,F). Global indices outside
    the domain map to local 0, exactly as at evaluation time, so verdicts match the logged rate."""
    from dig2.data.unified.dataset import (
        SEMANTIC_EDGE_MAPS, SEMANTIC_EDGE_VOCAB_SIZE, SEMANTIC_NODE_VOCAB, SEMANTIC_NODE_VOCAB_SIZE,
        _SEMANTIC_NODE_MAPS_BASE)
    domain = str(domain).strip().lower()
    node_maps = {**_SEMANTIC_NODE_MAPS_BASE, "qm9": [2, 3, 4, 5] if remove_h else [1, 2, 3, 4, 5]}
    if domain not in node_maps or domain not in SEMANTIC_EDGE_MAPS:
        raise KeyError(f"no semantic map for domain {domain!r}")
    node_map, edge_map = node_maps[domain], SEMANTIC_EDGE_MAPS[domain]
    atom_remap = torch.zeros(SEMANTIC_NODE_VOCAB_SIZE, dtype=torch.long)
    for local_idx, global_idx in enumerate(node_map):
        atom_remap[global_idx] = local_idx
    edge_remap = torch.zeros(SEMANTIC_EDGE_VOCAB_SIZE, dtype=torch.long)
    for local_idx, global_idx in enumerate(edge_map):
        edge_remap[global_idx] = local_idx
    global_to_element = {v: k for k, v in SEMANTIC_NODE_VOCAB.items()}
    return atom_remap, edge_remap, [global_to_element[g] for g in node_map]


def molecular_remaps(domain, global_vocab, remove_h):
    """Remaps for decoding a samples file. Graphs in the global vocabulary get the global->local
    maps; graphs already in the domain's local vocabulary get identities of the local sizes
    (applying the global maps there would collapse every atom onto local index 0)."""
    from dig2.data.unified.dataset import SEMANTIC_EDGE_MAPS
    atom_remap, edge_remap, atom_decoder = semantic_remaps_for_domain(domain, remove_h=remove_h)
    if global_vocab:
        return atom_remap, edge_remap, atom_decoder
    n_edge_local = len(SEMANTIC_EDGE_MAPS[str(domain).strip().lower()])
    return (torch.arange(len(atom_decoder), dtype=torch.long),
            torch.arange(n_edge_local, dtype=torch.long), atom_decoder)


def panel_molecule_row(xg, Eg, remaps):
    """RDKit's DetectChemistryProblems verdict on the unsanitized molecule, with valence errors
    broken down into 4-bonded N, over-bonded C (>=5), hypervalent S (>=7) and other, plus the
    number of fragments -- says WHY a molecule is invalid, not just whether."""
    from rdkit import Chem
    from dig2.metrics.rdkit_functions import build_molecule
    ar, er, dec = remaps
    X = ar[xg.clamp(0, ar.numel() - 1)]
    E = er[Eg.clamp(0, er.numel() - 1)]
    m = build_molecule(X, E, dec)
    m = m.GetMol() if hasattr(m, "GetMol") else m
    probs = Chem.DetectChemistryProblems(m)
    n4 = c5 = s_hyper = other = 0
    for p in probs:
        if p.GetType() == "AtomValenceException":
            a = m.GetAtomWithIdx(p.GetAtomIdx())
            bo = sum(b.GetBondTypeAsDouble() for b in a.GetBonds())
            sym = a.GetSymbol()
            if sym == "N" and bo == 4:
                n4 = 1
            elif sym == "C" and bo >= 5:
                c5 = 1
            elif sym == "S" and bo >= 7:
                s_hyper = 1
            else:
                other = 1
        else:
            other = 1
    G = nx.Graph()
    G.add_nodes_from(range(m.GetNumAtoms()))
    G.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds())
    return {"valid": int(len(probs) == 0), "n4": n4, "c5": c5, "s_hyper": s_hyper, "other_problem": other,
            "n_frag": int(nx.number_connected_components(G)) if G.number_of_nodes() else 0}


def plain_molecule_verdict(X_raw, E_raw, atom_remap, edge_remap, atom_decoder):
    """`BasicMolecularMetrics.compute_validity` for one molecule: valid when the whole-molecule
    SMILES sanitizes AND GetMolFrags succeeds (fragments allowed). `smiles` is the largest
    fragment's SMILES, which can be None even for a valid molecule -- compute_validity counts it
    valid anyway, and so does this."""
    from rdkit import Chem
    from dig2.metrics.rdkit_functions import build_molecule, mol2smiles

    result = {"ok": False, "fail_reason": "build_error", "smiles": None, "n_fragments": None}
    atoms = torch.as_tensor(X_raw).long()
    edges = torch.as_tensor(E_raw).long()
    if atoms.numel() == 0:
        return result
    atoms = atom_remap[atoms.clamp(min=0, max=atom_remap.numel() - 1)]
    edges = edge_remap[edges.clamp(min=0, max=edge_remap.numel() - 1)]
    try:
        mol = build_molecule(atoms, edges, atom_decoder)
    except Exception:  # noqa: BLE001
        return result
    if mol2smiles(mol) is None:
        result["fail_reason"] = "sanitize_fail"
        return result
    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Chem.rdchem.AtomValenceException:
        result["fail_reason"] = "valence_error"
        return result
    except Chem.rdchem.KekulizeException:
        result["fail_reason"] = "kekulize_error"
        return result
    largest = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
    result.update(ok=True, fail_reason="pass", smiles=mol2smiles(largest), n_fragments=len(frags))
    return result


def canonical_train_smiles(path):
    """RDKit-canonical SMILES of a training split (.npy of SMILES), for novelty."""
    from rdkit import Chem
    can = set()
    for s in np.load(path, allow_pickle=True):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            can.add(Chem.MolToSmiles(m))
    return can


# --------------------------------------------------------------------------------- loaders

def load_samples(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def samples_schema(samples):
    """"dict" for the multi-domain sampler's [{"graph": [X, E], "dataset_name": ...}, ...],
    "list" for the single-domain sampler's [[X, E], ...]."""
    return "dict" if (samples and isinstance(samples[0], dict)) else "list"


def domain_items(samples, domain):
    """[(X int64, E int64), ...] of one domain, in file order. A list-schema file carries no
    domain label, so all of it is taken as `domain`."""
    out = []
    for it in samples:
        if isinstance(it, dict) and str(it.get("dataset_name", "")).lower() != domain:
            continue
        g = it["graph"] if isinstance(it, dict) else it
        out.append((np.asarray(g[0]).astype(np.int64), np.asarray(g[1]).astype(np.int64)))
    return out


def dense_tokens(sample):
    """(node tokens, dense n x n edge-token matrix) of one unified-dataset graph."""
    xg = sample.x_token.view(-1).long()
    n = int(xg.numel())
    E = torch.zeros(n, n, dtype=torch.long)
    ei = sample.edge_index.long()
    if ei.numel():
        E[ei[0], ei[1]] = sample.edge_token.view(-1).long()
    return xg, E


def read_run_config(run_dir):
    """The run's recorded Hydra config as a plain dict, or None."""
    p = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(p):
        return None
    import yaml
    with open(p) as f:
        return yaml.safe_load(f) or {}
