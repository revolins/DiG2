"""Thin wrapper around the official MOSES benchmarking package (`molsets`, imported as
`moses`) exposing the four metrics DeFoG reports for MOSES/Guacamol-style molecular
generation benchmarking (2410.04263v3, Tab. 9/10): Filters, Frechet ChemNet Distance
(FCD), SNN (similarity to nearest neighbor), and Scaf (scaffold similarity).

Kept isolated from `metrics/molecular_metrics.py` so the rest of the codebase never
imports `moses` directly -- this module owns both the import-time compatibility shims
(see `_ensure_moses_compat_shims`) needed to run the (unmaintained) `molsets` package
against a modern rdkit/pandas, and the SMILES-list plumbing.

Install: `pip install --no-deps git+https://github.com/molecularsets/moses.git` (the
package is published under the import name `moses` but there is an unrelated PyPI
package literally called `moses`, so it must be installed from GitHub) plus
`pip install fcd_torch` (declared as a real dependency by molsets' setup.py, skipped by
--no-deps). `--no-deps` avoids molsets' `pomegranate` dependency, which is only needed
for its own generative baselines (HMM), not for `moses.metrics`.
"""

import sys
import types
import warnings

_SHIMS_APPLIED = False


def _ensure_moses_compat_shims():
    """Patch two known incompatibilities between `molsets` and modern rdkit/pandas.

    `molsets` (last released 2021) has not been updated for:
    1. `rdkit.six` was removed from rdkit years ago; `moses/metrics/SA_Score/sascorer.py`
       still does `from rdkit.six import iteritems`.
    2. `pandas.DataFrame.append` was removed in pandas 2.0; `moses/metrics/utils.py`
       uses it at import time to concatenate its MCF + PAINS filter SMARTS tables.

    Both shims are no-ops if the real APIs already exist (e.g. a future rdkit/pandas
    reintroduces them, or an older pinned version is used instead).
    """
    global _SHIMS_APPLIED
    if _SHIMS_APPLIED:
        return

    if "rdkit.six" not in sys.modules:
        try:
            import rdkit.six  # noqa: F401  (already available, nothing to shim)
        except ImportError:
            shim = types.ModuleType("rdkit.six")
            shim.iteritems = lambda d: iter(d.items())
            sys.modules["rdkit.six"] = shim

    import pandas as pd

    if not hasattr(pd.DataFrame, "append"):
        def _append(self, other, ignore_index=False, sort=False, **kwargs):
            return pd.concat([self, other], ignore_index=ignore_index, sort=sort)

        pd.DataFrame.append = _append

    _SHIMS_APPLIED = True


def _canonicalize(smiles_list):
    """Drop Nones/empties and strip whitespace.

    Some `dataset_smiles` sources hand back lines read straight from a `.smiles` file
    (e.g. `moses_dataset.py`'s `filter_dataset=True` path uses `open(...).readlines()`)
    which keeps trailing "\\n" characters; RDKit's SMILES parser silently rejects those,
    which would otherwise quietly shrink the reference set.
    """
    return [s.strip() for s in (smiles_list or []) if s and s.strip()]


def compute_moses_metrics(
    generated_smiles,
    reference_smiles_by_split,
    n_jobs=1,
    device="cpu",
    batch_size=512,
    precomputed_test_stats=None,
):
    """Compute MOSES-style Filters/FCD/SNN/Scaf for a set of generated molecules.

    Args:
        generated_smiles: list of SMILES strings (may contain None for invalid
            molecules; these are filtered out, matching `compute_fcd()`'s existing
            convention in `metrics/molecular_metrics.py`).
        reference_smiles_by_split: dict with keys "train" (required), "test"
            (required), and optionally "test_scaffolds" (the MOSES scaffold-holdout
            split; only `moses_dataset.py` currently exposes this -- see
            `get_smiles()` there). When absent, the Test-split values are reused for
            the TestSF-suffixed keys.
        n_jobs, device, batch_size: forwarded to `moses.metrics.metrics.get_all_metrics`.

    Returns:
        dict with the four headline keys DeFoG's Tab. 9/10 report --
        "Filters", "FCD", "SNN", "Scaf" (FCD/SNN use the Test split, Scaf uses TestSF
        when available, matching the convention used by DiGress/DisCo/Cometh/DeFoG
        when reporting a single MOSES-metrics column) -- plus every raw key
        `get_all_metrics` returns (e.g. "FCD/Test", "FCD/TestSF", "SNN/TestSF",
        "IntDiv", "Novelty", ...) so nothing is silently discarded.

        Returns {"Filters": -1, "FCD": -1, "SNN": -1, "Scaf": -1} (headline keys only)
        if `moses` is not importable or the computation raises, mirroring the existing
        `compute_fcd()` failure convention of logging and returning a sentinel instead
        of crashing the eval loop.
    """
    headline_fallback = {"Filters": -1, "FCD": -1, "SNN": -1, "Scaf": -1}

    gen = _canonicalize(generated_smiles)
    train = _canonicalize(reference_smiles_by_split.get("train"))
    test = _canonicalize(reference_smiles_by_split.get("test"))
    test_scaffolds = _canonicalize(reference_smiles_by_split.get("test_scaffolds")) or None

    if len(gen) == 0 or len(test) == 0:
        print("compute_moses_metrics: empty generated or test SMILES list, skipping.")
        return dict(headline_fallback)

    try:
        _ensure_moses_compat_shims()
        from moses.metrics.metrics import get_all_metrics
    except Exception as e:
        print(f"compute_moses_metrics: could not import moses ({e}). Skipping.")
        return dict(headline_fallback)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            # `precomputed_test_stats` (moses' compute_intermediate_statistics output) skips the
            # reference pre-calculation, which is ~5 ms/molecule and dominates cost on large
            # references. It is passed for BOTH ptest and ptest_scaffolds because
            # band_reference_smiles hands the same list object to "test" and "test_scaffolds" --
            # so today every cell pays that precalc TWICE over identical data. Passing it
            # reproduces existing numbers exactly (get_all_metrics would have called the same
            # deterministic function itself) -- but only on device="cpu"; a CUDA ChemNet forward
            # is nondeterministic at ~1e-6, which would desync cached vs fresh stats.
            _extra = {}
            if precomputed_test_stats is not None:
                if str(device) != "cpu":
                    raise ValueError(
                        "precomputed_test_stats requires device='cpu': cached and freshly-computed "
                        "ChemNet stats can differ at ~1e-6 on CUDA, which would make dual-reference "
                        "FCD deltas irreproducible against already-reported numbers.")
                _extra = {"ptest": precomputed_test_stats,
                          "ptest_scaffolds": precomputed_test_stats}
            raw = get_all_metrics(
                gen=gen,
                k=None,
                n_jobs=n_jobs,
                device=device,
                batch_size=batch_size,
                test=test,
                test_scaffolds=test_scaffolds,
                train=train,
                **_extra,
            )
        except Exception as e:
            print(f"compute_moses_metrics: get_all_metrics failed ({e}). Skipping.")
            return dict(headline_fallback)

    out = dict(raw)
    out["Filters"] = raw.get("Filters", -1)
    out["FCD"] = raw.get("FCD/Test", -1)
    out["SNN"] = raw.get("SNN/Test", -1)
    out["Scaf"] = raw.get("Scaf/TestSF", raw.get("Scaf/Test", -1))
    return out
