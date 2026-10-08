"""Size-band reference sets for discrete domain interpolation, in the shared semantic vocabulary.

Interpolated generations are scored zero-shot against GuacaMol or MOSES molecules. Two
conversions are needed and both live here, so the encoder and the decoder can never drift:

  * ``smiles_to_semantic_tokens`` -- a reference SMILES into the model's GLOBAL semantic
    vocabulary (``dig2.data.unified.dataset``), bypassing every per-domain local vocabulary.
    This is what lets a held-out set be evaluated WITHOUT registering it as a sixth domain,
    which would change the conditioning one-hot width and break strict loading.
  * ``semantic_dense_to_smiles`` -- a GENERATED graph, which is already in the global semantic
    vocabulary, back to canonical SMILES. ``dig2.metrics.rdkit_functions.build_molecule`` is
    keyed on a per-domain atom decoder and cannot decode it.

Representation decisions:
  * Heavy atoms only (implicit H); explicit H would push MOSES from a median of 22 atoms to 40,
    out of range for both molecular domains.
  * Kekulised (``clearAromaticFlags=True``); the global edge vocabulary has no aromatic class.
  * Atoms outside {H, C, N, O, F, P, S, Cl, Br, I} are rejected (0% of MOSES, 0.47% of GuacaMol).

References are compared band by band: a whole-dataset reference spanning 4-67 heavy atoms
scored against size-14 generations measures the SIZE mismatch, not chemistry. The two sources
are never pooled within a band -- the fraction of molecules using only QM9's {C,N,O,F} is
near-flat across GuacaMol's bands but swings from 0% to 63% across MOSES's.

Usage (prebuilding the parse caches is optional; they are built on first use):

    python -m dig2.tools.smiles_bands --sources guacamol,moses --cache-dir outputs/domain_interp/refcache
    python -m dig2.tools.smiles_bands --report --bands 15_19,20_27
"""
import argparse
import os

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

from dig2 import paths
from dig2.data.unified.dataset import (
    SEMANTIC_EDGE_VOCAB, SEMANTIC_EDGE_VOCAB_SIZE, SEMANTIC_NODE_VOCAB, SEMANTIC_NODE_VOCAB_SIZE,
)

RDLogger.DisableLog("rdApp.*")

# Bumping this invalidates every on-disk parse cache.
CONVERTER_VERSION = "dig2_semantic_smiles/1"

SX = SEMANTIC_NODE_VOCAB_SIZE
SE = SEMANTIC_EDGE_VOCAB_SIZE

SEMANTIC_BOND_TO_IDX = {
    BT.SINGLE: SEMANTIC_EDGE_VOCAB["SINGLE"],
    BT.DOUBLE: SEMANTIC_EDGE_VOCAB["DOUBLE"],
    BT.TRIPLE: SEMANTIC_EDGE_VOCAB["TRIPLE"],
}
INV_SEMANTIC_NODE_VOCAB = {v: k for k, v in SEMANTIC_NODE_VOCAB.items()}
_IDX_TO_BOND = {
    SEMANTIC_EDGE_VOCAB["SINGLE"]: BT.SINGLE,
    SEMANTIC_EDGE_VOCAB["DOUBLE"]: BT.DOUBLE,
    SEMANTIC_EDGE_VOCAB["TRIPLE"]: BT.TRIPLE,
}

# QM9's semantic node vocabulary under remove_h=true.
QM9_VOCAB_GLOBAL = {SEMANTIC_NODE_VOCAB[a] for a in ("C", "N", "O", "F")}

# Size bands in heavy atoms. QM9 spans 2-9 and ZINC 8-37, so band index is the axis along which
# an interpolated model is asked to leave both training distributions.
SIZE_BANDS = [(2, 9), (10, 14), (15, 19), (20, 27), (28, 37), (38, 999)]
# Representative node count per band, for the size-forced arm (mid-band, integer).
BAND_SIZES = {"2_9": 8, "10_14": 12, "15_19": 17, "20_27": 23, "28_37": 32, "38p": 45}


def band_name(lo, hi):
    return f"{lo}_{hi}" if hi < 999 else f"{lo}p"


def band_bounds(band):
    """``'15_19' -> (15, 19)``; ``'38p' -> (38, 999)``."""
    band = str(band).strip()
    if band.endswith("p"):
        return int(band[:-1]), 999
    lo, hi = band.split("_")
    return int(lo), int(hi)


# --------------------------------------------------------------------------------------
# SMILES <-> semantic tokens
# --------------------------------------------------------------------------------------

def smiles_to_semantic_tokens(smiles, keep_h=False):
    """``(record, reject_reason)``; ``reject_reason == ""`` on success.

    ``record = {x_token: LongTensor[n], edge_index: LongTensor[2, 2E], edge_token: LongTensor[2E],
    n_nodes: int, n_h: int, smiles: str}``.

    Reject reasons: ``parse_fail | kekulize_fail | oov_atom:<sym> | aromatic_bond_survived |
    no_bonds``.
    """
    # Sanitised parse (the default): skipping sanitisation skips ring perception, which makes
    # Kekulize fail unpredictably.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "parse_fail"

    # Count implicit + explicit H before stripping, as a hydrogen-content covariate.
    n_h = sum(a.GetTotalNumHs() + (1 if a.GetSymbol() == "H" else 0) for a in mol.GetAtoms())

    # RDKit's SMILES parser folds bracket [H] into implicit counts, so explicit hydrogens have
    # to be re-materialised with AddHs rather than merely "not removed".
    mol = Chem.AddHs(mol) if keep_h else Chem.RemoveHs(mol)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None, "kekulize_fail"

    type_idx = []
    for atom in mol.GetAtoms():
        gid = SEMANTIC_NODE_VOCAB.get(atom.GetSymbol())
        if gid is None:
            return None, f"oov_atom:{atom.GetSymbol()}"
        type_idx.append(gid)

    n = len(type_idx)
    if n == 0:
        return None, "no_bonds"

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        gid = SEMANTIC_BOND_TO_IDX.get(bond.GetBondType())
        if gid is None:
            # Post-kekulisation this is an invariant violation, not a routine reject.
            return None, "aromatic_bond_survived"
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [s, e]
        col += [e, s]
        edge_type += 2 * [gid]

    if len(row) == 0:
        return None, "no_bonds"

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_token = torch.tensor(edge_type, dtype=torch.long)
    # The canonical ordering the molecular loaders use, so token streams are comparable.
    perm = (edge_index[0] * n + edge_index[1]).argsort()
    return {
        "x_token": torch.tensor(type_idx, dtype=torch.long),
        "edge_index": edge_index[:, perm],
        "edge_token": edge_token[perm],
        "n_nodes": n,
        "n_h": int(n_h),
        "smiles": smiles,
    }, ""


def semantic_dense_to_smiles(x_idx, e_idx, sanitize=True):
    """Decode a GENERATED graph in the global semantic vocabulary to canonical SMILES.

    ``x_idx``: 1-D int tensor/array of global node ids, length n.
    ``e_idx``: ``(n, n)`` int tensor/array of global edge ids (0 = no edge).

    Returns ``(smiles, reason)``; ``smiles`` is None on failure and ``reason`` names it. A
    structural ('generic') node or edge class makes the graph non-molecular and is reported as
    such rather than silently dropped.
    """
    try:
        xs = [int(v) for v in (x_idx.tolist() if hasattr(x_idx, "tolist") else x_idx)]
        n = len(xs)
        m = Chem.RWMol()
        for t in xs:
            sym = INV_SEMANTIC_NODE_VOCAB.get(int(t))
            if sym is None or sym == "generic":
                return None, f"non_molecular_node:{t}"
            m.AddAtom(Chem.Atom(sym))
        E = e_idx.tolist() if hasattr(e_idx, "tolist") else e_idx
        for i in range(n):
            for j in range(i + 1, n):
                b = int(E[i][j])
                if b == SEMANTIC_EDGE_VOCAB["no_edge"]:
                    continue
                bt = _IDX_TO_BOND.get(b)
                if bt is None:
                    return None, f"non_molecular_edge:{b}"
                m.AddBond(i, j, bt)
        mol = m.GetMol()
        if sanitize:
            Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol), ""
    except Exception as e:
        return None, f"{type(e).__name__}"


# --------------------------------------------------------------------------------------
# reference sources
# --------------------------------------------------------------------------------------

def data_root(data_dir=None):
    """Absolute dataset directory: ``data_dir`` if absolute, else relative to the repo root."""
    d = str(data_dir or "data")
    return d if os.path.isabs(d) else os.path.join(str(paths.repo_root()), d)


def default_cache_dir():
    return os.path.join(str(paths.repo_root()), "outputs", "domain_interp", "refcache")


def _guacamol_raw(data_dir, fname):
    return os.path.join(data_root(data_dir), "guacamol", "guacamol_pyg", "raw", fname)


def _moses_raw(data_dir, fname):
    return os.path.join(data_root(data_dir), "moses", "moses_pyg", "raw", fname)


# (source, split) -> (path builder, reader). The held-out ("test") split is GuacaMol's official
# validation file and MOSES's released val file, which is MOSES's scaffold split -- the same
# convention the molecular metrics use.
_SOURCES = {
    ("guacamol", "train"): (lambda d: _guacamol_raw(d, "guacamol_v1_train.smiles"), "smiles"),
    ("guacamol", "test"): (lambda d: _guacamol_raw(d, "guacamol_v1_valid.smiles"), "smiles"),
    ("moses", "train"): (lambda d: _moses_raw(d, "train_moses.csv"), "csv"),
    ("moses", "test"): (lambda d: _moses_raw(d, "val_moses.csv"), "csv"),
}

_REF_MEMO = {}


def _read_source(source, split, data_dir=None):
    if (source, split) not in _SOURCES:
        raise ValueError(f"unknown reference source/split {source!r}/{split!r}; "
                         f"known: {sorted(_SOURCES)}")
    build, kind = _SOURCES[(source, split)]
    path = build(data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Point --data-dir at the directory holding the raw "
            f"{source} files, or let dig2.train download them first.")
    if kind == "smiles":
        from dig2.data.guacamol import extract_smiles_from_file
        return [s for s in extract_smiles_from_file(path) if s], path
    from dig2.data.moses import extract_smiles_from_csv
    return [str(s) for s in extract_smiles_from_csv(path) if str(s).strip()], path


def parsed_source(source, split, cache_dir=None, data_dir=None):
    """Parse a whole reference split ONCE into ``{smiles, n_nodes, qm9_vocab}``, memoised in
    process and cached on disk.

    Band filtering is applied to the parsed result, so a sweep over bands and alphas parses
    GuacaMol's 1.3M training molecules once rather than once per cell.
    """
    key = (source, split)
    if key in _REF_MEMO:
        return _REF_MEMO[key]

    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, f"ref__{source}__{split}.pt")
    if os.path.exists(cpath):
        try:
            payload = torch.load(cpath, map_location="cpu", weights_only=False)
            if payload.get("converter_version") == CONVERTER_VERSION:
                _REF_MEMO[key] = payload
                print(f"[interp] reference cache hit: {os.path.basename(cpath)} "
                      f"({len(payload['smiles'])} molecules)", flush=True)
                return payload
        except Exception:
            pass

    smis, src_path = _read_source(source, split, data_dir)
    print(f"[interp] parsing {source}/{split} once ({len(smis)} SMILES from {src_path}) -- "
          f"cached hereafter", flush=True)

    keep_s, keep_n, keep_v = [], [], []
    for smi in smis:
        rec, _why = smiles_to_semantic_tokens(smi, keep_h=False)
        if rec is None:
            continue
        keep_s.append(rec["smiles"])
        keep_n.append(int(rec["n_nodes"]))
        keep_v.append(bool(set(rec["x_token"].tolist()) <= QM9_VOCAB_GLOBAL))
    payload = {"converter_version": CONVERTER_VERSION, "source": source, "split": split,
               "source_file": os.path.abspath(src_path), "n_source": len(smis),
               "smiles": keep_s,
               "n_nodes": torch.tensor(keep_n, dtype=torch.int16),
               "qm9_vocab": torch.tensor(keep_v, dtype=torch.bool)}
    try:
        torch.save(payload, cpath)
        print(f"[interp] wrote reference cache {cpath}", flush=True)
    except Exception as e:
        print(f"[interp] WARNING: could not write {cpath} ({type(e).__name__})", flush=True)
    _REF_MEMO[key] = payload
    return payload


def band_reference_smiles(band_lo, band_hi, source="guacamol", qm9_vocab_only=False,
                          cache_dir=None, data_dir=None, with_train=True):
    """Band-matched reference SMILES in the split layout the MOSES-style metrics expect.

    Returns ``{"train": [...], "test": [...], "test_scaffolds": [...]}``. ``train`` feeds Novelty
    only; ``with_train=False`` skips parsing it (the training split is the expensive one) and
    returns an empty list for it.
    """
    def _sel(split):
        p = parsed_source(source, split, cache_dir=cache_dir, data_dir=data_dir)
        n = p["n_nodes"].numpy()
        m = (n >= band_lo) & (n <= band_hi)
        if qm9_vocab_only:
            m = m & p["qm9_vocab"].numpy()
        return [p["smiles"][i] for i in m.nonzero()[0]]

    tr = _sel("train") if with_train else []
    te = _sel("test")
    return {"train": tr, "test": te, "test_scaffolds": te}


def band_size_histogram(band_lo, band_hi, source="guacamol", qm9_vocab_only=False,
                        split="test", cache_dir=None, data_dir=None):
    """Empirical node-count distribution of the band-matched reference itself, as ``(ns, probs)``.

    The target-prior arm draws its generated sizes from this, so the generated set reproduces
    the reference's size DISTRIBUTION instead of pinning every molecule to one mid-band integer
    (a delta spike unlike any real reference, which puts a floor under FCD). Uses the same band
    filter and source as ``band_reference_smiles``, so prior and reference cannot drift apart.
    """
    p = parsed_source(source, split, cache_dir=cache_dir, data_dir=data_dir)
    n = p["n_nodes"].numpy().astype(np.int64)
    m = (n >= band_lo) & (n <= band_hi)
    if qm9_vocab_only:
        m = m & p["qm9_vocab"].numpy()
    sel = n[m]
    if sel.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    ns, cnt = np.unique(sel, return_counts=True)
    return ns, cnt.astype(np.float64) / float(cnt.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="guacamol,moses")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--data-dir", default=None,
                    help="dataset directory holding the raw guacamol/moses files "
                         "(default: the repo's data/)")
    ap.add_argument("--cache-dir", default=None,
                    help=f"where the parse caches go (default: {default_cache_dir()})")
    ap.add_argument("--report", action="store_true",
                    help="print the per-band population and vocabulary composition of each "
                         "parsed split instead of only building the cache")
    ap.add_argument("--bands", default=",".join(band_name(lo, hi) for lo, hi in SIZE_BANDS))
    a = ap.parse_args()

    sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    bands = [b.strip() for b in a.bands.split(",") if b.strip()]

    for src in sources:
        for split in splits:
            p = parsed_source(src, split, cache_dir=a.cache_dir, data_dir=a.data_dir)
            print(f"[interp] {src}/{split}: {len(p['smiles'])} usable of {p['n_source']} "
                  f"source SMILES", flush=True)
            if not a.report:
                continue
            for b in bands:
                lo, hi = band_bounds(b)
                nn = p["n_nodes"].numpy()
                m = (nn >= lo) & (nn <= hi)
                frac = float(p["qm9_vocab"].numpy()[m].mean()) if m.any() else float("nan")
                print(f"[interp]   band {b:>6}: n={int(m.sum()):7d}  qm9-vocab {frac:6.1%}",
                      flush=True)


if __name__ == "__main__":
    main()
