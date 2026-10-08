"""Shared machinery for discrete domain interpolation between two molecular domains.

The model conditions a graph on its domain through two learned objects:

  S1  ``domain_emb_table[idx]`` -> residual MLP -> conditioning vector
      (``GraphDiscreteInterpolativeModel._inject_domain_embedding``)
  S2  ``domain_prototypes[argmax(y_t)]`` -> ``struct_fusion([proto, current_emb])``
      (``_structural_proto_for_y`` / ``_build_structural_cond``)

Both are plain row lookups keyed by the domain index of the conditioning one-hot, which is
pinned to the destination domain for the whole sweep. So the interpolation is done by
temporarily overwriting the looked-up ROW of both objects and then running completely
unmodified model code -- every call site is covered, and alpha = 0 / alpha = 1 are bitwise
identical to the plain hard-lookup path by construction (the endpoints clone the endpoint row
instead of computing ``(1-a)*x + a*y``).

Blending only S1 would leave the structural conditioning pinned to one domain while alpha
moves, which silently measures a half-blend: ``domain_prototypes`` is not an analysis object,
it is a live conditioning input.

Library module -- no CLI. The driver is ``dig2.tools.domain_interp``.
"""
import contextlib
import os

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dig2 import paths
from dig2.checkpoint import load_weights_strict, merge_embedded_config, override_keys, read_checkpoint

# Semantic vocabulary widths of the shared feature schema (dig2.data.unified.dataset).
SX = 11
SE = 5


# --------------------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------------------

def compose_config(overrides=None, scratch_dir=None, ckpt_path=None):
    """Compose DiG2's config with Hydra overrides, as ``dig2.tools.export_ckpt`` does.

    When ``ckpt_path`` is a release checkpoint (one carrying an embedded ``dig2_config``), the
    architecture it records is overlaid on the defaults (explicit ``overrides`` still win), so the
    model is rebuilt with the settings its weights were trained with.
    """
    with initialize_config_dir(
        config_dir=str(paths.repo_root() / "dig2" / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="dig2", overrides=list(overrides or []))
    OmegaConf.set_struct(cfg, False)
    if ckpt_path is not None:
        cfg = merge_embedded_config(cfg, ckpt_path, keep=override_keys(overrides))
    if scratch_dir is not None:
        os.makedirs(scratch_dir, exist_ok=True)
        # Real training sets this from the active Hydra run; point it at a scratch directory so
        # nothing written during interpolation lands next to a training run's outputs.
        cfg.hier_dir = scratch_dir
    return cfg


def attach_trainer(model, datamodule, device):
    """Wire ``model.trainer.datamodule`` without running a fit/validate loop.

    ``_to_dense_maybe_tokenized`` and ``_sample_domain_num_nodes`` read
    ``self.trainer.datamodule``; outside ``trainer.fit/validate/test`` ``self.trainer`` raises.
    ``attach_data`` is the same low-level call Lightning's own entry points use.
    """
    import pytorch_lightning as pl

    trainer = pl.Trainer(
        accelerator="gpu" if torch.device(device).type == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer._data_connector.attach_data(model, datamodule=datamodule)
    return trainer


def build_and_load(ckpt_path, overrides=None, scratch_dir=None, device=None,
                   strict_asserts=True, compute_reference=False):
    """Build the model from a composed config, strict-load ``ckpt_path``, attach a Trainer.

    Returns ``(model, datamodule, cfg, device, ckpt_path)``.
    """
    from dig2.pipeline import build_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    cfg = compose_config(overrides, scratch_dir=scratch_dir, ckpt_path=ckpt_path)
    model, datamodule = build_model(cfg, compute_reference=compute_reference)
    n = load_weights_strict(model, ckpt_path)
    print(f"[interp] strict load OK: {n} tensors from {ckpt_path}", flush=True)
    attach_trainer(model, datamodule, device)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    if strict_asserts:
        assert_conditioning_contract(model, datamodule, cfg)
    report_prototype_norms(model, datamodule)
    return model, datamodule, cfg, device, ckpt_path


def checkpoint_has_embedded_config(ckpt_path) -> bool:
    """True for a release checkpoint (``dig2.tools.export_ckpt`` output)."""
    _, cfg = read_checkpoint(ckpt_path)
    return cfg is not None


def domain_index_map(model, datamodule=None):
    """``{domain name: index}`` as the model itself resolves it (never a restated constant)."""
    idx = dict(model._domain_name_to_index())
    if not idx and datamodule is not None:
        idx = {str(n): i for i, n in enumerate(datamodule.dataset_names)}
    return {str(k).strip().lower(): int(v) for k, v in idx.items()}


def report_prototype_norms(model, datamodule=None):
    """Print the per-domain prototype norms. Diagnostic only -- no expectation is enforced: the
    EMA writes a centred, L2-normalised mean, so the norms depend on the trained checkpoint and
    there is no fixed reference value to compare against."""
    if not hasattr(model, "domain_prototypes"):
        return {}
    P = model.domain_prototypes.detach().cpu().float()
    got = {n: float(P[i].norm()) for n, i in domain_index_map(model, datamodule).items()
           if i < P.size(0)}
    print(f"[interp] prototype norms: { {k: round(v, 4) for k, v in got.items()} }", flush=True)
    return got


def assert_conditioning_contract(model, datamodule, cfg):
    """Fail loudly if this is not the conditioning topology the interpolation assumes.

    A strict weight load already proves the architecture matches the checkpoint; these checks
    prove the *conditioning topology* is the one the row override covers.
    """
    def _chk(cond, msg):
        if not cond:
            raise AssertionError(f"[interp] conditioning contract violated: {msg}")

    idx = domain_index_map(model, datamodule)
    n_dom = len(datamodule.dataset_names)
    emb_dim = int(cfg.general.domain_emb_dim)

    _chk(getattr(model, "domain_emb_table", None) is not None,
         "model has no domain_emb_table (general.domain_embedding must be true)")
    _chk(tuple(model.domain_emb_table.weight.shape) == (n_dom + 1, emb_dim),
         f"domain_emb_table {tuple(model.domain_emb_table.weight.shape)} "
         f"!= ({n_dom + 1}, {emb_dim}) -- n_domains + 1 rows (the learned CFG null) expected")
    _chk(int(getattr(model, "domain_null_idx", -1)) == n_dom,
         f"domain_null_idx={getattr(model, 'domain_null_idx', None)}, expected {n_dom}")
    _chk(bool(getattr(model, "structural_encoder_enabled", False)),
         "the structural encoder is disabled, so domain_prototypes carries no conditioning")
    _chk(tuple(model.domain_prototypes.shape) == (n_dom, emb_dim),
         f"domain_prototypes {tuple(model.domain_prototypes.shape)} != ({n_dom}, {emb_dim})")
    _chk(float(model.domain_prototypes.abs().sum()) > 0,
         "domain_prototypes are all zero (untrained prototype bank)")
    # A second embedding table inside the transformer would make the row override a half-blend.
    _chk(getattr(model.model, "domain_emb", None) is None,
         "model.model.domain_emb is not None -- a second domain embedding table is active and "
         "the row override would cover only part of the conditioning")

    gv = getattr(datamodule, "global_vocab", None)
    _chk(isinstance(gv, dict) and int(gv.get("Sx", 0)) == SX and int(gv.get("Se", 0)) == SE,
         f"global_vocab Sx/Se != {SX}/{SE}")
    order = [str(d).strip().lower() for d in cfg.dataset.unified.domain_order]
    _chk([str(d).strip().lower() for d in datamodule.dataset_names] == order,
         f"dataset_names={list(datamodule.dataset_names)} != domain_order={order}")
    for d in ("qm9", "zinc"):
        _chk(d in idx, f"domain {d!r} is absent from this model's domain map {sorted(idx)}")
    print("[interp] conditioning contract OK", flush=True)


# --------------------------------------------------------------------------------------
# the interpolation
# --------------------------------------------------------------------------------------

@contextlib.contextmanager
def blended_domain_conditioning(model, alpha, src="qm9", dst="zinc",
                                blend_embedding=True, blend_prototype=True, verbose=False,
                                datamodule=None):
    """Set row[dst] of BOTH domain tables to ``(1-a) * row[src] + a * row[dst]``, then restore.

    The conditioning one-hot stays pinned at ``dst`` for the whole sweep, so every consumer that
    does ``table[argmax(y)]`` picks up the blended row with no change to model code. ``alpha``
    may exceed 1 (extrapolation past the destination domain).

    alpha == 0.0 / 1.0 short-circuit to an exact clone of the endpoint row, so the tensors handed
    to the model are BITWISE identical to the unblended lookup (this also removes the
    ``(1-a)*x + a*y == -0.0`` corner case).
    """
    idx = domain_index_map(model, datamodule)
    i_s, i_d = idx[str(src).strip().lower()], idx[str(dst).strip().lower()]
    a = float(alpha)

    def _lerp(W):
        if a == 0.0:
            return W[i_s].detach().clone()
        if a == 1.0:
            return W[i_d].detach().clone()
        return ((1.0 - a) * W[i_s] + a * W[i_d]).detach().clone()

    saved_W = model.domain_emb_table.weight.detach().clone()
    saved_P = model.domain_prototypes.detach().clone()

    touched = []
    try:
        if blend_embedding:
            model.domain_emb_table.weight.data[i_d] = _lerp(saved_W)
            touched.append("domain_emb_table")
        if blend_prototype:
            model.domain_prototypes.data[i_d] = _lerp(saved_P)
            touched.append("domain_prototypes")
        if verbose:
            print(f"[interp] alpha={a:.6g}: blended rows {touched} ({src} -> {dst})", flush=True)
        yield
    finally:
        model.domain_emb_table.weight.data.copy_(saved_W)
        model.domain_prototypes.data.copy_(saved_P)


@contextlib.contextmanager
def union_vocab_mask(model, domains=("qm9", "zinc"), verbose=False, datamodule=None):
    """Install the union of the two domains' vocabularies on both of their rows, held constant
    across alpha.

    Without it, the destination domain's hard vocabulary mask sends every atom type the other
    domain does not use to -1e9, so the generated support would be a function of the mask rather
    than of the conditioning.
    """
    from dig2.model.helpers import domain as domain_helpers

    saved = getattr(model, "_Vd_mask_cache", None)
    saved_dev = getattr(model, "_Vd_mask_device_cache", None)
    Vx, Ve = domain_helpers._domain_vocab_valid_masks(model, Sx=SX, Se=SE, device=None)
    if Vx is None:
        raise AssertionError("[interp] semantic V_d masks unavailable; no union mask can be built")
    Vx, Ve = Vx.clone(), Ve.clone()
    idx = domain_index_map(model, datamodule)
    ids = [idx[str(d).strip().lower()] for d in domains]
    ux, ue = Vx[ids].any(0), Ve[ids].any(0)
    e_changed = not all(bool(torch.equal(Ve[i], ue)) for i in ids)
    for i in ids:
        Vx[i] = ux
        Ve[i] = ue
    if verbose:
        print(f"[interp] union vocab X={ux.nonzero().flatten().tolist()} "
              f"E={ue.nonzero().flatten().tolist()} (edge support changed: {e_changed})",
              flush=True)
    try:
        model._Vd_mask_cache = (Vx, Ve)
        # The helper memoises device-resident copies in a SEPARATE dict, keyed by device and
        # vocabulary width, and a hit there short-circuits before ``_Vd_mask_cache`` is
        # consulted. Clearing it is what makes this override -- and its restoration -- actually
        # take effect instead of reusing whichever masks were installed first in the process.
        model._Vd_mask_device_cache = {}
        yield {"union_x": ux.nonzero().flatten().tolist(),
               "union_e": ue.nonzero().flatten().tolist(),
               "edge_support_changed": e_changed}
    finally:
        model._Vd_mask_cache = saved
        model._Vd_mask_device_cache = dict(saved_dev or {})


def one_hot_domain(name, batch_size, device, model=None, datamodule=None, n_domains=None):
    """``[batch_size, n_domains]`` conditioning one-hot pinned to one domain."""
    idx = domain_index_map(model, datamodule)
    if n_domains is None:
        n_domains = len(datamodule.dataset_names) if datamodule is not None else len(idx)
    y = torch.zeros(int(batch_size), int(n_domains), device=device)
    y[:, idx[str(name).strip().lower()]] = 1.0
    return y
