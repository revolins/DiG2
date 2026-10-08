"""Generate molecules with interpolated domain conditioning and score them zero-shot.

The conditioning rows of the two molecular domains are blended as
``(1 - alpha) * qm9 + alpha * zinc`` (``dig2.tools.domain_interp_common``), generation is
conditioned on ZINC, and the union of the two domains' vocabularies is allowed. Generated
molecules are scored against GuacaMol or MOSES molecules OF THE SAME SIZE BAND
(``dig2.tools.smiles_bands``). ``alpha`` may exceed 1 to extrapolate past ZINC.

Size is a knob to PIN, not to sweep: QM9 spans 2-9 heavy atoms and ZINC 8-37, so a blended
size prior is a MIXTURE -- it moves the mean while staying bimodal, and a metric that moves
with it is reading size, not chemistry. Hence three arms, which answer different questions:

  ``size_forced``   every graph is pinned to the band's mid-band size at every alpha, so size
                    is constant across alpha by construction. Supports a chemical claim, but
                    its delta-spike size distribution is unlike any real reference and puts a
                    floor under FCD.
  ``target_prior``  sizes are drawn from the BAND REFERENCE's own node-count histogram, and the
                    same size vector is reused at every alpha. Size is both target-matched in
                    distribution and byte-identical across alpha, so the alpha cells are a
                    PAIRED design. This is the arm to use for a chemical claim.
  ``free_running``  the blended per-domain size prior chooses. Measures whether alpha controls
                    molecule SIZE at all, which is a separate question.

Metric feasibility is decided from the actual sample sizes, not from the band name: FCD is a
Frechet distance between fits of ~512-d ChemNet activations, so a covariance estimated from
n molecules has rank at most n-1. Below the rank floor the value is not a measurement and is
suppressed; between the floor and the stability threshold it is reported with a note that it
is biased upward.

Usage:
    python -m dig2.tools.domain_interp --ckpt checkpoints/dig2.ckpt \
        --arm target_prior --bands 15_19 --alphas 0.0,0.5,1.0,1.5 \
        --n-samples 2000 --seed 0 --crn --ref-source guacamol \
        --save-smiles-dir outputs/domain_interp/smiles \
        --out-json outputs/domain_interp/downstream.json

    # endpoint check: alpha=0 / alpha=1 must reproduce plain qm9 / plain zinc conditioning
    python -m dig2.tools.domain_interp --ckpt checkpoints/dig2.ckpt --endpoint-check
"""
import argparse
import contextlib
import json
import os
import time
import zlib

import numpy as np
import torch

from dig2.tools import smiles_bands as sb
from dig2.tools.domain_interp_common import (
    blended_domain_conditioning, build_and_load, one_hot_domain, union_vocab_mask,
)

# A reference below this size cannot support a stable 512-d FCD covariance.
FCD_MIN_REFERENCE = 1000
# The same argument applies to the generated side: 512 is the ChemNet activation
# dimensionality (the rank floor), 1000 the stability threshold.
FCD_GENERATED_RANK_FLOOR = 512
FCD_MIN_GENERATED = 1000
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


@contextlib.contextmanager
def blended_node_count_prior(model, alpha, src="qm9", dst="zinc"):
    """Blend the two domains' node-count histograms, by cache override (free_running arm).

    ``_sample_domain_num_nodes`` memoises ``(hist, min_n, max_n)`` per domain in
    ``self._cached_domain_hist``, so writing the blended entry there lets unmodified model code
    do the sampling. The blended support must span BOTH domains: keeping the source domain's
    max_n would clamp every draw to <= 9 even at alpha = 1 and silently defeat the blend.
    """
    def _entry(name):
        model._sample_domain_num_nodes(name, 1)           # populates the cache the normal way
        return (getattr(model, "_cached_domain_hist", {}) or {}).get(str(name))

    e_s, e_d = _entry(src), _entry(dst)
    cache = getattr(model, "_cached_domain_hist", None)
    if cache is None:
        model._cached_domain_hist = cache = {}
    saved = dict(cache)
    try:
        if e_s is None or e_d is None:
            print(f"[interp] WARNING: no node-count histogram for {src!r}/{dst!r}; this arm "
                  "falls back to the UNBLENDED prior and does not measure what it claims",
                  flush=True)
        else:
            hs, mins, maxs = e_s
            hd, mind, maxd = e_d
            n = max(hs.numel(), hd.numel())
            a_ = torch.zeros(n, device=hs.device)
            a_[: hs.numel()] = hs
            b_ = torch.zeros(n, device=hd.device)
            b_[: hd.numel()] = hd
            mix = (1.0 - alpha) * a_ + alpha * b_
            mix = mix / mix.sum().clamp(min=1e-12)
            cache[str(dst)] = (mix, min(mins, mind), max(maxs, maxd))
        yield
    finally:
        model._cached_domain_hist = saved


def generate_batch(model, *, batch_size, num_nodes, forced_condition):
    """One sampling call. ``number_chain_steps`` must be >= 1 (``sample_batch`` computes
    ``1 / number_chain_steps``); ``keep_chain=0`` already makes the chain tensors zero-width,
    so 1 costs nothing and retains nothing."""
    return model.sample_batch(
        batch_id=0, batch_size=batch_size, keep_chain=0, number_chain_steps=1,
        save_final=0, num_nodes=num_nodes, save_visualization=False,
        forced_condition=forced_condition)


def to_smiles(samples):
    """``(valid_smiles, rejects, valid_sizes)``.

    ``valid_sizes`` is the node count of the graphs that SURVIVED decoding, aligned 1:1 with
    ``valid_smiles``. It is reported separately from the all-generated size mean because
    decoding failure is size-dependent, so the scored subset can be systematically smaller than
    the set that was drawn -- a bias that matching sample sizes does not remove, only expose.
    """
    out, rejects, vsizes = [], {}, []
    for s in samples:
        try:
            x, e = (s[0], s[1]) if isinstance(s, (list, tuple)) else (s["graph"][0], s["graph"][1])
            xi = torch.as_tensor(x).long().view(-1)
            ei = torch.as_tensor(e).long()
            if ei.dim() == 1:
                ei = ei.view(xi.numel(), xi.numel())
            smi, why = sb.semantic_dense_to_smiles(xi, ei)
            n_i = int(xi.numel())
        except Exception as ex:
            smi, why, n_i = None, type(ex).__name__, None
        if smi:
            out.append(smi)
            vsizes.append(n_i)
        else:
            rejects[why] = rejects.get(why, 0) + 1
    return out, rejects, vsizes


def _graph_fingerprint(samples):
    """Byte-level fingerprint of a batch of generated graphs."""
    import hashlib

    h = hashlib.sha256()
    for x, e in ((s[0], s[1]) for s in samples):
        xt = torch.as_tensor(x).to(torch.int64).contiguous()
        et = torch.as_tensor(e).to(torch.int64).contiguous()
        h.update(str(tuple(xt.shape)).encode())
        h.update(xt.numpy().tobytes())
        h.update(str(tuple(et.shape)).encode())
        h.update(et.numpy().tobytes())
    return h.hexdigest()


def endpoint_check(model, datamodule, device, *, batch_size, num_nodes, seed, src, dst,
                   epsilon=1e-6):
    """The endpoints must be exact, and an epsilon must not be.

    What is checked, and why each form is the right one:

    * ROWS. alpha = 0 must hand the model bitwise the source domain's conditioning row (in both
      the embedding table and the prototype bank) and alpha = 1 bitwise the destination's. This
      is the property the row override guarantees by construction, and the only one that is
      independent of everything else the domain index selects.
    * OUTPUTS at alpha = 1. Conditioning on the destination domain with its own rows restored is
      the plain destination path, so the generated graphs must be bitwise identical to it.
    * OUTPUTS at alpha = 0 are NOT required to equal the plain source path, and with
      ``model.prior_mode = domain_specific`` they do not: the limit (noise) distribution and the
      guidance weight are selected by the conditioning one-hot, which stays pinned at the
      DESTINATION domain for the whole sweep. That is deliberate -- a prior that moved with
      alpha would change the sampling problem at every alpha -- but it means alpha = 0 is "the
      source domain's conditioning under the destination domain's prior", not the source domain.
      The equality does hold under ``model.prior_mode=default`` (one global prior), which is how
      this claim is testable rather than assumed.
    * An epsilon away from an endpoint must change the conditioning rows, otherwise the dial is
      not connected to anything.
    """
    from dig2.tools.domain_interp_common import domain_index_map

    idx = domain_index_map(model, datamodule)
    i_s, i_d = idx[str(src).strip().lower()], idx[str(dst).strip().lower()]
    nn = torch.full((batch_size,), int(num_nodes), dtype=torch.long, device=device)
    prior_mode = str(getattr(model.cfg.model, "prior_mode", "")).strip().lower()

    def _run(alpha, cond_domain):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        row = idx[str(cond_domain).strip().lower()]
        ctx = contextlib.nullcontext() if alpha is None else blended_domain_conditioning(
            model, alpha, src=src, dst=dst, datamodule=datamodule)
        with ctx, union_vocab_mask(model, (src, dst), datamodule=datamodule):
            # The rows the model will actually look up for this conditioning one-hot.
            consumed = (model.domain_emb_table.weight[row].detach().cpu().clone(),
                        model.domain_prototypes[row].detach().cpu().clone())
            forced = one_hot_domain(cond_domain, batch_size, device, model=model,
                                    datamodule=datamodule)
            samples, _ = generate_batch(model, batch_size=batch_size, num_nodes=nn,
                                        forced_condition=forced)
        return _graph_fingerprint(samples), consumed

    plain_src, rows_src = _run(None, src)
    plain_dst, rows_dst = _run(None, dst)
    a0, rows_a0 = _run(0.0, dst)
    a1, rows_a1 = _run(1.0, dst)
    aeps, rows_eps = _run(epsilon, dst)

    def _eq(r1, r2):
        return bool(torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1]))

    res = {
        "batch_size": int(batch_size), "num_nodes": int(num_nodes), "seed": int(seed),
        "src": src, "dst": dst, "epsilon": float(epsilon), "prior_mode": prior_mode,
        "fingerprints": {f"plain_{src}": plain_src, f"plain_{dst}": plain_dst,
                         "alpha_0": a0, "alpha_1": a1, f"alpha_{epsilon:g}": aeps},
        "alpha0_rows_equal_src_rows": _eq(rows_a0, rows_src),
        "alpha1_rows_equal_dst_rows": _eq(rows_a1, rows_dst),
        "alpha1_output_equals_plain_dst": a1 == plain_dst,
        "epsilon_rows_differ_from_dst_rows": not _eq(rows_eps, rows_dst),
        # Informational: required only when one global prior is in use (see the docstring).
        "alpha0_output_equals_plain_src": a0 == plain_src,
        "alpha0_output_equality_required": prior_mode != "domain_specific",
        "epsilon_output_differs_from_plain_dst": aeps != plain_dst,
    }
    required = ["alpha0_rows_equal_src_rows", "alpha1_rows_equal_dst_rows",
                "alpha1_output_equals_plain_dst", "epsilon_rows_differ_from_dst_rows"]
    if res["alpha0_output_equality_required"]:
        required.append("alpha0_output_equals_plain_src")
    res["required_checks"] = required
    res["passed"] = all(bool(res[k]) for k in required)

    print("[interp] endpoint check:", flush=True)
    for k in required:
        print(f"[interp]   {k}: {'PASS' if res[k] else 'FAIL'}", flush=True)
    for k in ("alpha0_output_equals_plain_src", "epsilon_output_differs_from_plain_dst"):
        print(f"[interp]   {k}: {res[k]}  (informational; "
              f"prior_mode={prior_mode})", flush=True)
    for k, v in res["fingerprints"].items():
        print(f"[interp]   sha256[{k}] = {v}", flush=True)
    print(f"[interp] endpoint check {'PASSED' if res['passed'] else 'FAILED'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to interpolate, strict-loaded (default: "
                         "checkpoints/dig2.ckpt). Always explicit about which epoch: a "
                         "wrong-epoch checkpoint produces perfectly plausible numbers labelled "
                         "with the wrong weights.")
    ap.add_argument("--override", action="append", default=[],
                    help="Hydra override for the architecture/sampling config, repeatable "
                         "(e.g. --override +experiment=readout_3_6_9_12)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--out-dir", default=None,
                    help="base directory for scratch, caches and outputs "
                         "(default: outputs/domain_interp under the repo root)")
    ap.add_argument("--cache-dir", default=None,
                    help="reference parse caches (default: <out-dir>/refcache)")
    ap.add_argument("--data-dir", default=None,
                    help="dataset directory with the raw guacamol/moses files "
                         "(default: the run config's dataset.datadir)")
    ap.add_argument("--src-domain", default="qm9")
    ap.add_argument("--dst-domain", default="zinc")
    ap.add_argument("--arm", default="target_prior",
                    choices=["size_forced", "free_running", "target_prior"])
    ap.add_argument("--bands", default=",".join(sb.BAND_SIZES))
    ap.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--target-valid", type=int, default=None,
                    help="match EFFECTIVE sample size instead of drawn sample size: keep "
                         "drawing until this many molecules DECODE, then score exactly this "
                         "many. FCD's bias scales with the number of VALID molecules, so "
                         "comparing cells of unequal validity at a fixed --n-samples reads that "
                         "bias as chemistry.")
    ap.add_argument("--max-generated", type=int, default=None,
                    help="hard cap on draws per cell under --target-valid (default: 20x "
                         "--target-valid). A cell that hits the cap is recorded with "
                         "target_valid_met=false and must not be compared against one that "
                         "met it -- that is the exact bias --target-valid removes.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None,
                    help="replicate seed; re-seeds each (band, alpha) cell independently. "
                         "Omit to leave the global RNG untouched.")
    ap.add_argument("--crn", action="store_true",
                    help="common random numbers: seed identically at every alpha within a "
                         "(seed, band) cell. With --arm target_prior (whose size vector is "
                         "already alpha-independent) the alpha cells then share both their "
                         "sizes and their initial noise, so a difference between two alphas is "
                         "attributable to conditioning with far less variance. Variance "
                         "reduction, not a bias.")
    ap.add_argument("--ref-source", default="guacamol", choices=["guacamol", "moses"])
    ap.add_argument("--qm9-vocab-only", action="store_true",
                    help="score against the vocabulary-matched control reference")
    ap.add_argument("--no-moses-metrics", action="store_true",
                    help="skip the MOSES-style metrics (Filters/FCD/SNN/Scaf) and the training "
                         "split they need; validity, uniqueness and sizes are still reported. "
                         "For smoke runs.")
    ap.add_argument("--save-smiles-dir", default=None,
                    help="write every cell's generated SMILES to "
                         "<dir>/<model>__<source>__<band>__a<alpha>.smi, one per line")
    ap.add_argument("--model-tag", default=None,
                    help="name used in the saved SMILES filenames (default: the checkpoint's "
                         "file stem)")
    ap.add_argument("--tag", default=None, help="suffix for the saved SMILES filenames")
    ap.add_argument("--out-json", default=None,
                    help="default: <out-dir>/domain_interp.json")
    ap.add_argument("--endpoint-check", action="store_true",
                    help="run the endpoint gate (alpha=0 / alpha=1 must reproduce plain source "
                         "/ destination conditioning bitwise) and exit")
    ap.add_argument("--endpoint-nodes", type=int, default=12)
    ap.add_argument("--endpoint-batch", type=int, default=4)
    args = ap.parse_args()

    from dig2 import paths

    out_dir = args.out_dir or os.path.join(str(paths.repo_root()), "outputs", "domain_interp")
    cache_dir = args.cache_dir or os.path.join(out_dir, "refcache")
    out_json = args.out_json or os.path.join(out_dir, "domain_interp.json")
    scratch_dir = os.path.join(out_dir, "_scratch")

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    src, dst = args.src_domain, args.dst_domain

    if args.target_valid is not None:
        _cap = args.max_generated or (20 * args.target_valid)
        if _cap < args.target_valid:
            ap.error("--max-generated must be >= --target-valid")
        _plan_len = _cap
        print(f"[interp] target-valid mode: scoring exactly {args.target_valid} valid molecules "
              f"per cell, drawing at most {_cap}", flush=True)
    else:
        _cap = _plan_len = args.n_samples

    ckpt_path = args.ckpt or str(paths.default_checkpoint())
    if not os.path.exists(ckpt_path):
        ap.error(f"checkpoint {ckpt_path} does not exist; pass --ckpt")
    model, dm, cfg, device, ckpt = build_and_load(
        ckpt_path, overrides=args.override, scratch_dir=scratch_dir, device=args.device)
    data_dir = args.data_dir or str(cfg.dataset.datadir)
    model_tag = args.model_tag or os.path.splitext(os.path.basename(str(ckpt)))[0]

    if args.endpoint_check:
        res = endpoint_check(model, dm, device, batch_size=args.endpoint_batch,
                             num_nodes=args.endpoint_nodes, seed=(args.seed or 0),
                             src=src, dst=dst)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump({"ckpt": os.path.abspath(str(ckpt)), "endpoint_check": res}, f, indent=2)
        print(f"[interp] wrote {out_json}", flush=True)
        raise SystemExit(0 if res["passed"] else 1)

    results = {}
    for band in bands:
        lo, hi = sb.band_bounds(band)
        print(f"[interp] building the band-matched reference for {band} from "
              f"{args.ref_source} ...", flush=True)
        ref = sb.band_reference_smiles(lo, hi, source=args.ref_source,
                                       qm9_vocab_only=args.qm9_vocab_only,
                                       cache_dir=cache_dir, data_dir=data_dir,
                                       with_train=not args.no_moses_metrics)
        n_ref = len(ref["test"])
        print(f"[interp]   reference n={n_ref} (train={len(ref['train'])})", flush=True)
        if n_ref == 0:
            print(f"[interp]   SKIP {band}: {args.ref_source} has no molecules in this band",
                  flush=True)
            continue

        # target_prior: ONE size vector per band, drawn from the reference's own histogram and
        # reused verbatim at every alpha. Seeded from (seed, band) and deliberately NOT from
        # alpha, which makes the alpha cells a PAIRED design -- byte-identical sizes, not merely
        # the same distribution.
        size_plan = None
        ref_size_mean = ref_size_std = None
        if args.arm == "target_prior":
            ns, ps = sb.band_size_histogram(lo, hi, source=args.ref_source,
                                            qm9_vocab_only=args.qm9_vocab_only,
                                            cache_dir=cache_dir, data_dir=data_dir)
            if ns.size == 0:
                print(f"[interp]   SKIP {band}: empty size histogram", flush=True)
                continue
            _rng = np.random.default_rng(zlib.crc32(f"{args.seed}|{band}|sizeplan".encode()))
            # Draw the FULL cap length once, so every alpha slices a prefix of the SAME array.
            # Under --target-valid the alphas consume different numbers of draws, and drawing
            # per-alpha at each cell's own length would rely on numpy's choice() being
            # prefix-stable -- an implementation detail, not a contract.
            size_plan = _rng.choice(ns, size=_plan_len, p=ps).astype(np.int64)
            ref_size_mean = float((ns * ps).sum())
            ref_size_std = float(np.sqrt((ps * (ns - ref_size_mean) ** 2).sum()))
            print(f"[interp]   arm=target_prior size plan: mean={size_plan.mean():.2f} "
                  f"sd={size_plan.std():.2f} range=[{size_plan.min()},{size_plan.max()}]  "
                  f"| reference mean={ref_size_mean:.2f} sd={ref_size_std:.2f}", flush=True)

        results[band] = {"n_reference": n_ref, "ref_source": args.ref_source,
                         "ref_size_mean": ref_size_mean, "ref_size_std": ref_size_std,
                         "alphas": {}}
        for a in alphas:
            n_nodes = None
            ctx = [blended_domain_conditioning(model, a, src=src, dst=dst, datamodule=dm,
                                               verbose=True),
                   union_vocab_mask(model, (src, dst), datamodule=dm)]
            if args.arm == "size_forced":
                n_nodes = torch.full((args.batch_size,), sb.BAND_SIZES[band],
                                     dtype=torch.long, device=device)
            elif args.arm == "free_running":
                ctx.append(blended_node_count_prior(model, a, src=src, dst=dst))

            if args.seed is not None:
                # Re-seed per cell so each (band, alpha) is independently reproducible and
                # replicate seeds stay independent instead of continuing one correlated stream.
                # zlib.crc32, not hash(): Python randomises str hashing per process, so
                # hash()-derived seeds do not reproduce across runs. Under --crn the alpha term
                # is dropped, so every alpha starts from the same RNG state.
                _cell = (f"{args.seed}|{band}|crn" if args.crn
                         else f"{args.seed}|{band}|{a}").encode()
                _sd = zlib.crc32(_cell) % (2 ** 31)
                torch.manual_seed(_sd)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(_sd)

            smiles, sizes, valid_sizes, rejects = [], [], [], {}
            with contextlib.ExitStack() as stack:
                for c in ctx:
                    stack.enter_context(c)
                n_done = 0
                bi = 0
                _t0 = time.time()
                while n_done < _cap:
                    if args.target_valid is not None and len(smiles) >= args.target_valid:
                        break
                    bs = min(args.batch_size, _cap - n_done)
                    if size_plan is not None:
                        nn = torch.as_tensor(size_plan[n_done:n_done + bs],
                                             dtype=torch.long, device=device)
                    elif n_nodes is not None:
                        nn = n_nodes[:bs]
                    else:
                        # Draw through _sample_domain_num_nodes -- the function
                        # blended_node_count_prior overrides. sample_batch(num_nodes=None) does
                        # NOT consult the per-domain prior: it falls back to the GLOBAL node
                        # distribution, so the blend would have no effect at all.
                        nn = model._sample_domain_num_nodes(dst, bs)
                        if nn is None:
                            print("[interp] WARNING: _sample_domain_num_nodes returned None; "
                                  "falling back to the GLOBAL node distribution -- this arm "
                                  "then does NOT measure the blended per-domain prior",
                                  flush=True)
                        else:
                            nn = nn.to(device)
                    forced = one_hot_domain(dst, bs, device, model=model, datamodule=dm)
                    samples, _labels = generate_batch(
                        model, batch_size=bs, num_nodes=nn, forced_condition=forced)
                    s, r, vs = to_smiles(samples)
                    smiles.extend(s)
                    valid_sizes.extend(vs)
                    for k, v in r.items():
                        rejects[k] = rejects.get(k, 0) + v
                    for x in samples:
                        try:
                            sizes.append(int(torch.as_tensor(x[0]).view(-1).numel()))
                        except Exception:
                            pass
                    n_done += bs
                    bi += 1
                    if args.target_valid is not None and bi % 10 == 0:
                        _el = time.time() - _t0
                        _rate = n_done / max(_el, 1e-9)
                        _need = max(args.target_valid - len(smiles), 0)
                        _vr = len(smiles) / max(n_done, 1)
                        _eta = (_need / _vr / _rate) if (_vr > 0 and _rate > 0) else float("nan")
                        print(f"[interp]   .. band={band} a={a:.2f} drawn={n_done}/{_cap} "
                              f"valid={len(smiles)}/{args.target_valid} ({_vr:.3f}) "
                              f"{_rate:.1f} samples/s eta~{_eta / 60:.1f}min", flush=True)

            # Validity is computed from the RAW counts, before any truncation: after trimming to
            # --target-valid the ratio is target/n_done, an artifact of the stopping rule.
            n_valid_raw = len(smiles)
            valid_frac = n_valid_raw / max(n_done, 1)
            target_met = None
            if args.target_valid is not None:
                target_met = n_valid_raw >= args.target_valid
                if target_met:
                    smiles = smiles[:args.target_valid]
                    valid_sizes = valid_sizes[:args.target_valid]
                else:
                    print(f"[interp]   WARNING band={band} alpha={a:.2f}: only {n_valid_raw} "
                          f"valid in {n_done} draws (cap), short of --target-valid "
                          f"{args.target_valid}. This cell's FCD carries the small-sample bias "
                          f"and is NOT comparable to cells that met the target.", flush=True)

            if args.save_smiles_dir:
                os.makedirs(args.save_smiles_dir, exist_ok=True)
                _fp = os.path.join(
                    args.save_smiles_dir,
                    f"{model_tag}__{args.ref_source}__{band}__a{a:.2f}"
                    + (f"__{args.tag}" if args.tag else "") + ".smi")
                with open(_fp, "w") as _f:
                    _f.write("\n".join(smiles) + ("\n" if smiles else ""))
                print(f"[interp]   wrote {len(smiles)} smiles -> {_fp}", flush=True)

            uniq = len(set(smiles)) / max(len(smiles), 1)
            met = {"Filters": -1, "FCD": -1, "SNN": -1, "Scaf": -1}
            if smiles and not args.no_moses_metrics:
                from dig2.metrics.moses_style import compute_moses_metrics

                met = compute_moses_metrics(smiles, ref)
                if n_ref < FCD_MIN_REFERENCE:
                    # Refuse to report a number that would look real.
                    met["FCD_raw_unreliable"] = met.get("FCD")
                    met["FCD"] = -1
                    met["FCD_note"] = (f"suppressed: reference n={n_ref} < "
                                       f"{FCD_MIN_REFERENCE}, too few for a stable 512-d FCD "
                                       "covariance")
                n_gen = len(smiles)
                if met.get("FCD", -1) != -1 and n_gen <= FCD_GENERATED_RANK_FLOOR:
                    met["FCD_raw_unreliable"] = met.get("FCD")
                    met["FCD"] = -1
                    met["FCD_note"] = (
                        f"suppressed: generated n={n_gen} <= {FCD_GENERATED_RANK_FLOOR}, "
                        "rank-deficient 512-d covariance -- not a measurement")
                    print(f"[interp]   FCD SUPPRESSED band={band} alpha={a:.2f}: only {n_gen} "
                          f"valid molecules, at or below the {FCD_GENERATED_RANK_FLOOR}-d rank "
                          "floor", flush=True)
                elif met.get("FCD", -1) != -1 and n_gen < FCD_MIN_GENERATED:
                    met["FCD_low_n"] = n_gen
                    met["FCD_note"] = (
                        f"defined but biased UPWARD: generated n={n_gen} < "
                        f"{FCD_MIN_GENERATED}; compare only against cells of similar n")
                    print(f"[interp]   FCD LOW-n band={band} alpha={a:.2f}: n={n_gen} < "
                          f"{FCD_MIN_GENERATED}, biased upward", flush=True)

            results[band]["alphas"][str(a)] = {
                "n_generated": n_done, "n_valid": len(smiles), "n_valid_raw": n_valid_raw,
                "target_valid": args.target_valid, "target_valid_met": target_met,
                "validity": valid_frac, "uniqueness": uniq,
                "size_mean": float(np.mean(sizes)) if sizes else None,
                "size_std": float(np.std(sizes)) if sizes else None,
                # Size of the SCORED set. Under --target-valid this can drift from size_mean
                # even though the size PLAN is byte-identical across alpha, because decoding
                # failure is size-dependent.
                "valid_size_mean": float(np.mean(valid_sizes)) if valid_sizes else None,
                "valid_size_std": float(np.std(valid_sizes)) if valid_sizes else None,
                "rejects": rejects,
                "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in met.items()},
            }
            print(f"[interp]   band={band} alpha={a:.2f} valid={valid_frac:.3f} "
                  f"({n_valid_raw}/{n_done}) scored_n={len(smiles)} uniq={uniq:.3f} "
                  f"size={np.mean(sizes) if sizes else float('nan'):.1f} "
                  f"vsize={np.mean(valid_sizes) if valid_sizes else float('nan'):.1f} "
                  f"FCD={met.get('FCD')} SNN={met.get('SNN')} Scaf={met.get('Scaf')}",
                  flush=True)

    # In both pinned arms the generated size must not vary with alpha, and target_prior must
    # additionally match the reference's size distribution -- otherwise it is size_forced with
    # extra steps.
    if args.arm in ("size_forced", "target_prior"):
        for band, r in results.items():
            szs = [v["size_mean"] for v in r["alphas"].values() if v["size_mean"] is not None]
            if args.target_valid is not None:
                # The alphas consume DIFFERENT-LENGTH prefixes of one shared size plan, so their
                # realised means legitimately differ -- the paired property is prefix identity,
                # not equal means. Check the invariant that still applies.
                tol = 0.5
                bad = [f"a={k}:{v['size_mean']:.2f}" for k, v in r["alphas"].items()
                       if v["size_mean"] is not None and r.get("ref_size_mean") is not None
                       and abs(v["size_mean"] - r["ref_size_mean"]) > tol]
                if bad:
                    print(f"[interp] WARNING band={band}: the size plan drifted from the "
                          f"reference mean {r['ref_size_mean']:.2f} in {len(bad)} cell(s): "
                          f"{', '.join(bad)}", flush=True)
                vsz = [v.get("valid_size_mean") for v in r["alphas"].values()
                       if v.get("valid_size_mean") is not None]
                if vsz:
                    print(f"[interp] band={band}: SCORED-set size across alpha "
                          f"{min(vsz):.2f}..{max(vsz):.2f} (spread {max(vsz) - min(vsz):.2f}). "
                          "A non-zero spread is the validity-vs-size bias that matching sample "
                          "sizes does NOT remove; read any FCD difference against it.",
                          flush=True)
            elif szs and (max(szs) - min(szs)) > 1e-6:
                print(f"[interp] WARNING band={band}: the generated size varies across alpha "
                      f"({min(szs):.2f}..{max(szs):.2f}) -- this arm must hold size CONSTANT, "
                      "so any metric movement is not provably chemical", flush=True)
            if args.arm == "target_prior" and r.get("ref_size_mean") is not None:
                sds = [v["size_std"] for v in r["alphas"].values() if v["size_std"] is not None]
                if szs and abs(szs[0] - r["ref_size_mean"]) > 0.5:
                    print(f"[interp] WARNING band={band}: generated size mean {szs[0]:.2f} != "
                          f"reference {r['ref_size_mean']:.2f} -- the target prior did not take",
                          flush=True)
                if sds and r["ref_size_std"] > 0 and abs(sds[0] - r["ref_size_std"]) > 0.5:
                    print(f"[interp] WARNING band={band}: generated size sd {sds[0]:.2f} != "
                          f"reference {r['ref_size_std']:.2f} -- a shape mismatch, not just a "
                          "mean one", flush=True)

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"ckpt": os.path.abspath(str(ckpt)), "overrides": list(args.override),
                   "arm": args.arm, "src_domain": src, "dst_domain": dst,
                   "seed": args.seed, "crn": args.crn, "n_samples": args.n_samples,
                   "target_valid": args.target_valid, "max_generated": _cap,
                   "ref_source": args.ref_source, "qm9_vocab_only": args.qm9_vocab_only,
                   "moses_metrics": not args.no_moses_metrics,
                   "alphas": alphas, "bands": results}, f, indent=2)
    print(f"[interp] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
