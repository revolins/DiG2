"""Molecular property scoring, shared by property targeting and its evaluation.

Every property is defined exactly once, here, so the value a generation run targets and the
value it is measured against can never come from two different implementations.

Properties:
  qed   rdkit.Chem.QED.qed              -- drug-likeness composite, [0, 1]
  logp  rdkit.Chem.Crippen.MolLogP       -- Crippen logP
"""
import numpy as np


def _qed(m):
    from rdkit.Chem import QED
    return float(QED.qed(m))


def _logp(m):
    from rdkit.Chem import Crippen
    return float(Crippen.MolLogP(m))


# name -> (fn, human direction). Direction is documentation only: targeting aims at a value
# rather than maximising, so "better" is |value - target| for every property alike.
PROPERTIES = {
    "qed":  (_qed,  "higher is more drug-like"),
    "logp": (_logp, "higher is more lipophilic"),
}


def score_smiles(smiles, props=("qed", "logp")):
    """-> (values, n_parse_fail, n_score_fail).

    `values` is {prop: np.array} over the molecules that parsed AND scored. A molecule that parses
    but throws in one property scorer is dropped from THAT property only, so one pathological
    molecule cannot silently shorten every column. Counts are returned rather than warned so the
    caller can put them in the cell record.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    for p in props:
        if p not in PROPERTIES:
            raise KeyError(f"unknown property {p!r}; known: {sorted(PROPERTIES)}")

    out = {p: [] for p in props}
    n_parse_fail = 0
    n_score_fail = {p: 0 for p in props}
    for s in smiles:
        if not s:
            n_parse_fail += 1
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            n_parse_fail += 1
            continue
        for p in props:
            try:
                out[p].append(PROPERTIES[p][0](m))
            except Exception:
                n_score_fail[p] += 1
    return ({p: np.asarray(v, dtype=float) for p, v in out.items()},
            n_parse_fail, n_score_fail)


def cell_property_row(smiles, n_attempted, prop, target=None, tol=None,
                      reference_values=None):
    """One cell's property record.

    Validity is computed against `n_attempted` (what was REQUESTED), never against the length of
    the scored list -- a cell truncated to a target count would otherwise report a validity that is
    an artifact of the stopping rule.
    """
    vals, n_parse_fail, n_score_fail = score_smiles(smiles, props=(prop,))
    v = vals[prop]
    n_attempted = int(n_attempted)
    row = {
        "property": prop,
        "n_attempted": n_attempted,
        "n_scored": int(v.size),
        "validity": (float(v.size) / n_attempted) if n_attempted else None,
        "n_parse_fail": int(n_parse_fail),
        "n_score_fail": int(n_score_fail[prop]),
        "mean": float(v.mean()) if v.size else None,
        "sd": float(v.std(ddof=1)) if v.size > 1 else None,
        "iqr": (float(np.percentile(v, 75) - np.percentile(v, 25))
                if v.size > 1 else None),
    }
    if target is not None and v.size:
        t = float(target)
        row["target"] = t
        row["abs_err"] = abs(float(v.mean()) - t)
        # Straddle check: a bimodal set can nail the MEAN while controlling nothing. Report the
        # split so a small abs_err with a ~50/50 split is visibly not control.
        row["frac_above"] = float((v > t).mean())
        row["frac_below"] = float((v < t).mean())
        if tol is not None:
            row["tol"] = float(tol)
            row["hit_rate"] = float((np.abs(v - t) <= float(tol)).mean())
    if reference_values is not None and v.size:
        try:
            from scipy.stats import wasserstein_distance
            row["w1_to_reference"] = float(
                wasserstein_distance(v, np.asarray(reference_values, dtype=float)))
        except Exception:
            row["w1_to_reference"] = None
    return row
