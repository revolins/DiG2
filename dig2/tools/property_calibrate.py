"""Calibrate alpha as a dial for a molecular property, and gate whether it is one at all.

The model has no property-conditioning axis: alpha -- the blend of two domain conditioning rows
-- is its only continuous dial. So "reach a target property along the alpha path" is possible
only if alpha moves that property by more than the noise. This measures that per property and
returns KEEP or DROP, with the numeric rule fixed in advance so the verdict cannot be
renegotiated after the numbers are in. The gate can genuinely fail: if every property DROPs,
the result is "alpha is not a property dial", for almost no compute.

Inputs are the ``.smi`` files ``dig2.tools.domain_interp --save-smiles-dir`` writes, plus
(optionally) its ``--out-json`` for per-cell validity, which a ``.smi`` file cannot carry
because it holds only the molecules that already decoded.

Leakage: alpha*(target) is fit on the FIT seeds only, and when the targets come from the
model's own reachable range that range must come from the same fit seeds -- otherwise the
evaluation data has helped choose the targets it is then scored against. The fit/eval seed sets
are asserted disjoint rather than assumed so.

Usage:
    python -m dig2.tools.property_calibrate \
        --smiles-dir outputs/domain_interp/smiles \
        --interp-json 'outputs/domain_interp/downstream_s*.json' \
        --band 15_19 --props qed,logp --fit-seeds 3,4 --eval-seeds 0,1,2 \
        --out outputs/domain_interp/property/calibration.json

    # screen a single unseeded pilot run (no validity available, G4/G5 skipped)
    python -m dig2.tools.property_calibrate --smiles-dir outputs/domain_interp/smiles --screen
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from dig2.metrics.property import PROPERTIES, score_smiles

# The alpha group tolerates a leading minus: the driver writes alpha=-0.25 as "a-0.25.smi", and
# a `[0-9.]+` group would skip exactly the exterior-alpha cells, silently.
FNAME = re.compile(
    r"^(?P<model>.+?)__(?P<src>[a-z]+)__(?P<band>[0-9]+_[0-9]+|[0-9]+p)"
    r"__a(?P<alpha>-?[0-9.]+)(?:__(?P<tag>.+))?\.smi$")

# --- the gate, fixed in advance -------------------------------------------------------------
G1_SE_MULT = 8.0      # the range must be >= 8 standard errors at production n
G2_SD_FRAC = 0.25     # ... and >= 0.25 within-cell sd, an n-independent practical floor
G4_MIN_VALIDITY = 0.70
G5_SEED_FRAC = 6.0    # the cross-seed sd of a cell mean must be <= range / 6
PROD_N = 500          # the n each production cell will be scored at


def _seed_of(tag):
    """The driver is invoked with --tag s<seed>, so the seed rides in the tag."""
    if not tag:
        return None
    m = re.search(r"s(\d+)$", tag)
    return int(m.group(1)) if m else None


def collect(smiles_dir, band, src, props):
    """-> ``{(alpha, seed): {prop: np.array, '_n': int}}`` plus the files that were skipped."""
    cells, skipped = {}, []
    for path in sorted(glob.glob(os.path.join(smiles_dir, "*.smi"))):
        m = FNAME.match(os.path.basename(path))
        if not m:
            skipped.append((os.path.basename(path), "the name does not match the grammar"))
            continue
        if m.group("band") != band or m.group("src") != src:
            continue
        smis = [l.strip() for l in open(path) if l.strip()]
        if not smis:
            skipped.append((os.path.basename(path), "empty"))
            continue
        vals, n_parse_fail, _ = score_smiles(smis, props=props)
        cells[(float(m.group("alpha")), _seed_of(m.group("tag")))] = dict(
            vals, _n=len(smis), _parse_fail=n_parse_fail, _file=os.path.basename(path))
    return cells, skipped


def load_validity(json_glob, band):
    """``{(alpha, seed): validity}`` from the driver's out-jsons. Validity is a gate input (G4)
    and cannot be recovered from a ``.smi`` file."""
    out = {}
    for path in sorted(glob.glob(json_glob)) if json_glob else []:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        seed = d.get("seed")
        for a, cell in d.get("bands", {}).get(band, {}).get("alphas", {}).items():
            out[(float(a), seed)] = cell.get("validity")
    return out


def admit_alphas(cells, validity, seeds, min_validity):
    """The alphas eligible for the grid: validity >= the floor at EVERY fit seed.

    ADMISSION, not post-hoc rejection. Computing the range over all alphas and then asking
    whether validity held across it lets one broken exterior cell fail an otherwise healthy
    property. A cell the model cannot generate is not part of the reachable range; it is
    excluded from it, and reported as excluded.
    """
    all_a = sorted({a for (a, _) in cells})
    admitted, excluded = [], {}
    for a in all_a:
        vs = [validity.get((a, s)) for s in (sorted(seeds) if seeds else [None])]
        vs = [v for v in vs if v is not None]
        if not vs:
            admitted.append(a)          # unknown validity (screen mode) -> do not exclude
            continue
        if min(vs) >= min_validity:
            admitted.append(a)
        else:
            excluded[a] = {"min_validity": float(min(vs)),
                           "reason": "below the G4 validity floor"}
    return admitted, excluded


def summarize(cells, seeds, prop):
    """Per-alpha mean over the given seeds, weighted by the number scored."""
    per_a = {}
    for (a, s), c in cells.items():
        if seeds is not None and s not in seeds:
            continue
        v = c.get(prop)
        if v is None or v.size == 0:
            continue
        per_a.setdefault(a, []).append((v.mean(), v.size, s))
    alphas = sorted(per_a)
    means = []
    for a in alphas:
        rows = per_a[a]
        w = np.array([r[1] for r in rows], float)
        means.append(float(np.average([r[0] for r in rows], weights=w)))
    return np.array(alphas), np.array(means), per_a


def pooled_sd(cells, seeds, prop):
    num = den = 0.0
    for (a, s), c in cells.items():
        if seeds is not None and s not in seeds:
            continue
        v = c.get(prop)
        if v is None or v.size < 2:
            continue
        num += (v.size - 1) * v.var(ddof=1)
        den += (v.size - 1)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def longest_monotone(alphas, means):
    """Indices of the longest strictly monotone run. A unique alpha* inversion needs
    monotonicity; where the curve folds, the usable SEGMENT is reported rather than a range
    computed across the fold, which would overstate what is reachable."""
    best = (0, 1)
    for sign in (+1, -1):
        i = 0
        while i < len(means) - 1:
            j = i
            while j < len(means) - 1 and sign * (means[j + 1] - means[j]) > 0:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = max(j, i + 1)
    return best


def pchip(alphas, means):
    try:
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(alphas, means, extrapolate=False)
    except Exception:
        def lin(x):
            return np.interp(x, alphas, means, left=np.nan, right=np.nan)
        return lin


def invert(fn, alphas, target, n_grid=4001):
    """The alpha with ``fn(alpha) == target``, by dense scan on the monotone segment. Returns
    None when the target is outside the fitted span -- no extrapolation."""
    g = np.linspace(alphas[0], alphas[-1], n_grid)
    y = np.asarray(fn(g), dtype=float)
    ok = np.isfinite(y)
    if not ok.any():
        return None
    g, y = g[ok], y[ok]
    if target < min(y) or target > max(y):
        return None
    return float(g[int(np.argmin(np.abs(y - target)))])


def reference_targets(band, src, prop, quantiles, cache_dir=None, data_dir=None):
    """Targets read off the HELD-OUT band reference distribution, not off any model's reach.

    The default (``--targets-from reach``) places targets inside the model's own measured
    reachable range, which is sound against its own null but reads as "targets were chosen
    where this model can win" in a multi-model table. Reading them off the reference set makes
    the choice model-independent: it is the same pool every arm is already scored against. A
    target may then land OUTSIDE a model's reach -- that is the point: ``invert`` returns None
    and the cell is reported as out of reach rather than silently clamped to an endpoint.
    """
    from dig2.tools.smiles_bands import band_bounds, band_reference_smiles

    lo, hi = band_bounds(band)
    pool = band_reference_smiles(lo, hi, source=src, cache_dir=cache_dir, data_dir=data_dir,
                                 with_train=False)
    # Use the held-out split: it is the same pool the downstream metrics score against.
    if isinstance(pool, dict):
        pool = pool.get("test") or next(iter(pool.values()))
    pool = [str(x) for x in pool]
    vals, _, _ = score_smiles(pool, props=(prop,))
    v = vals[prop]
    if v.size == 0:
        raise SystemExit(f"[cal] the reference pool for band {band} scored 0 molecules "
                         f"on {prop!r}")
    qs = np.percentile(v, [q * 100.0 for q in quantiles])
    return ([float(np.round(t, 4)) for t in qs],
            {"n_reference": int(v.size), "quantiles": list(quantiles),
             "reference_mean": float(v.mean()),
             "reference_p": {f"p{int(q * 100)}": float(t) for q, t in zip(quantiles, qs)}})


def loo_floor(alphas, means):
    """Leave-one-alpha-out: refit without each interior alpha and predict it. The max/RMS error
    is the CALIBRATION FLOOR -- the best ``|mean - target|`` a cell can reach under perfect
    execution, so a miss smaller than the floor is not a miss."""
    errs = []
    for i in range(1, len(alphas) - 1):
        a_fit = np.delete(alphas, i)
        m_fit = np.delete(means, i)
        pred = pchip(a_fit, m_fit)(alphas[i])
        if np.isfinite(pred):
            errs.append(abs(float(pred) - means[i]))
    if not errs:
        return {"max": None, "rms": None, "per_alpha": []}
    return {"max": float(np.max(errs)), "rms": float(np.sqrt(np.mean(np.square(errs)))),
            "per_alpha": [float(e) for e in errs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles-dir", required=True)
    ap.add_argument("--interp-json", default=None,
                    help="glob of dig2.tools.domain_interp out-jsons, for validity (gate G4)")
    ap.add_argument("--band", default="15_19")
    ap.add_argument("--src", default="guacamol")
    ap.add_argument("--props", default="qed,logp")
    ap.add_argument("--fit-seeds", default="3,4")
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--screen", action="store_true",
                    help="unseeded pilot: use every cell as the fit set and SKIP G4/G5. "
                         "Produces a screen, never a verdict.")
    ap.add_argument("--targets-from", default="reach",
                    help="'reach' (default): targets at 20/50/80%% of the model's own measured "
                         "reachable range. 'reference[:p25,p50,p75]': targets at the given "
                         "quantiles of the held-out band reference distribution, i.e. "
                         "model-independent targets taken from the dataset.")
    ap.add_argument("--cache-dir", default=None,
                    help="reference parse caches, for --targets-from reference")
    ap.add_argument("--data-dir", default=None,
                    help="dataset directory with the raw guacamol/moses files")
    ap.add_argument("--out", default="outputs/domain_interp/property/calibration.json")
    args = ap.parse_args()

    tf = args.targets_from.strip()
    if tf == "reach":
        _tq = None
    elif tf == "reference" or tf.startswith("reference:"):
        spec = tf.split(":", 1)[1] if ":" in tf else "p25,p50,p75"
        try:
            _tq = [float(x.strip().lstrip("p")) / 100.0 for x in spec.split(",") if x.strip()]
        except ValueError:
            ap.error(f"could not parse quantiles from --targets-from {tf!r}")
        if not _tq or any(not (0.0 < q < 1.0) for q in _tq):
            ap.error(f"--targets-from quantiles must be in (0,100), got {spec!r}")
    else:
        ap.error("--targets-from must be 'reach' or 'reference[:p25,p50,p75]'")

    props = [p.strip() for p in args.props.split(",") if p.strip()]
    for p in props:
        if p not in PROPERTIES:
            ap.error(f"unknown property {p!r}; known: {sorted(PROPERTIES)}")

    fit = None if args.screen else {int(s) for s in args.fit_seeds.split(",") if s.strip()}
    ev = None if args.screen else {int(s) for s in args.eval_seeds.split(",") if s.strip()}
    if not args.screen:
        overlap = fit & ev
        if overlap:
            ap.error(f"LEAKAGE: the fit and eval seeds overlap on {sorted(overlap)}. alpha* "
                     "would be fit on the same data it is scored against.")

    cells, skipped = collect(args.smiles_dir, args.band, args.src, props)
    if not cells:
        ap.error(f"no {args.src} band-{args.band} .smi cells in {args.smiles_dir}")
    for f, why in skipped:
        print(f"[cal] skip {f}: {why}", flush=True)

    validity = load_validity(args.interp_json, args.band)
    admitted, excluded = admit_alphas(cells, validity, fit, G4_MIN_VALIDITY)
    for a, info in sorted(excluded.items()):
        print(f"[cal] EXCLUDE alpha={a}: validity {info['min_validity']:.3f} "
              f"< {G4_MIN_VALIDITY} -- measured, reported, and kept OUT of the reachable range")
    seeds_seen = sorted({s for (_, s) in cells}, key=lambda x: (x is None, x))
    print(f"[cal] {len(cells)} cells, alphas={sorted({a for a, _ in cells})}, seeds={seeds_seen}")
    if args.screen:
        print("[cal] SCREEN mode: all cells used as fit, G4/G5 skipped -- not a verdict.")

    out = {"band": args.band, "src": args.src, "screen": bool(args.screen),
           "fit_seeds": sorted(fit) if fit else None,
           "eval_seeds": sorted(ev) if ev else None,
           "gate": {"G1_se_mult": G1_SE_MULT, "G2_sd_frac": G2_SD_FRAC,
                    "G4_min_validity": G4_MIN_VALIDITY, "G5_seed_frac": G5_SEED_FRAC,
                    "prod_n": PROD_N},
           "properties": {}}

    for prop in props:
        alphas, means, per_a = summarize(cells, fit, prop)
        keep = np.array([a in admitted for a in alphas], dtype=bool)
        alphas, means = alphas[keep], means[keep]
        if alphas.size < 3:
            print(f"\n[{prop}] SKIP -- only {alphas.size} alpha cells in the fit set")
            continue
        s = pooled_sd(cells, fit, prop)
        se = s / np.sqrt(PROD_N)

        i0, i1 = longest_monotone(alphas, means)
        mono_full = (i1 - i0) == (len(alphas) - 1)
        a_seg, m_seg = alphas[i0:i1 + 1], means[i0:i1 + 1]
        R = float(abs(m_seg[-1] - m_seg[0]))

        # G4 holds by construction: only admitted alphas reach this point. The observed minimum
        # is reported so the margin is visible rather than implied.
        vals_seg = [validity.get((float(a), sd)) for a in a_seg for sd in (fit or {None})]
        vals_seg = [v for v in vals_seg if v is not None]
        g4 = (min(vals_seg) >= G4_MIN_VALIDITY) if vals_seg else None
        g4_min = float(min(vals_seg)) if vals_seg else None

        # G5: the cross-seed stability of each cell mean.
        seed_sds = [float(np.std([r[0] for r in per_a[a]], ddof=1))
                    for a in a_seg if len(per_a.get(a, [])) > 1]
        g5 = (max(seed_sds) <= R / G5_SEED_FRAC) if seed_sds else None

        g1, g2, g3 = R >= G1_SE_MULT * se, R >= G2_SD_FRAC * s, mono_full
        hard = [g1, g2]
        soft = [g for g in (g4, g5) if g is not None]
        verdict = "KEEP" if all(hard) and all(soft) else "DROP"

        fn = pchip(a_seg, m_seg)
        lo, hi = float(min(m_seg)), float(max(m_seg))
        if args.targets_from == "reach":
            targets = [float(np.round(lo + q * (hi - lo), 4)) for q in (0.20, 0.50, 0.80)]
            tgt_meta = {"source": "reach", "fractions": [0.20, 0.50, 0.80]}
        else:
            targets, tgt_meta = reference_targets(args.band, args.src, prop, _tq,
                                                  cache_dir=args.cache_dir,
                                                  data_dir=args.data_dir)
            tgt_meta["source"] = "reference"
        astar = {f"{t}": invert(fn, a_seg, t) for t in targets}
        # invert() refuses to extrapolate beyond the fitted span, so an unreachable target gives
        # alpha*=None. Record that explicitly -- a missing alpha* must never read as a 0.0.
        tgt_meta["reachable"] = {f"{t}": (astar[f"{t}"] is not None) for t in targets}
        floor = loo_floor(a_seg, m_seg)

        print(f"\n[{prop}] {verdict}")
        print(f"   alphas        {np.round(a_seg, 3).tolist()}")
        print(f"   means         {np.round(m_seg, 4).tolist()}")
        print(f"   range R       {R:.4f}     pooled sd s {s:.4f}    SE@n{PROD_N} {se:.4f}")
        print(f"   G1 R>={G1_SE_MULT}*SE  {g1}   ({R:.4f} vs {G1_SE_MULT * se:.4f}, "
              f"{R / se:.1f} SE)")
        print(f"   G2 R>={G2_SD_FRAC}*sd  {g2}   ({R:.4f} vs {G2_SD_FRAC * s:.4f})")
        print(f"   G3 monotone      {g3}" + ("" if mono_full else
              f"   -- using the segment alpha[{a_seg[0]:.2f},{a_seg[-1]:.2f}]"))
        print(f"   G4 validity      {g4}   (min over the admitted alphas: {g4_min})")
        print(f"   G5 seed-stable   {g5}")
        print(f"   targets ({tgt_meta['source']}) {targets}")
        print(f"   reachable        {tgt_meta['reachable']}")
        print(f"   alpha*           "
              f"{ {k: (round(v, 4) if v is not None else None) for k, v in astar.items()} }")
        print(f"   calib floor      max={floor['max']} rms={floor['rms']}")

        out["properties"][prop] = {
            "verdict": verdict, "R": R, "pooled_sd": s, "se_at_prod_n": float(se),
            "R_over_se": float(R / se) if se else None,
            "monotone_full": bool(mono_full),
            "usable_alpha_segment": [float(a_seg[0]), float(a_seg[-1])],
            "alphas": a_seg.tolist(), "means": m_seg.tolist(),
            "gates": {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": g4, "G5": g5},
            "g4_min_validity": g4_min,
            "excluded_alphas": {str(k): v for k, v in excluded.items()},
            "reachable_lo": lo, "reachable_hi": hi,
            "targets": targets, "alpha_star": astar, "targets_meta": tgt_meta,
            "calibration_floor": floor,
            "per_alpha_seed_means": {f"{a}": {str(r[2]): float(r[0]) for r in per_a[a]}
                                     for a in a_seg},
        }

    kept = [p for p, d in out["properties"].items() if d["verdict"] == "KEEP"]
    print(f"\n[cal] KEEP: {kept or 'NONE'}")
    if not kept:
        print("[cal] Every property failed the gate. alpha is not a property dial at this "
              "band; report the measured ranges and spend no generation compute on targeting.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[cal] wrote {args.out}")


if __name__ == "__main__":
    main()
