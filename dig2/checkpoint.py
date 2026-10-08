"""Checkpoint loading.

Two formats load the same way:
  * a Lightning checkpoint written by `dig2.train` (``state_dict`` + ``hyper_parameters``);
  * a release checkpoint written by `dig2.tools.export_ckpt` (``state_dict`` + ``dig2_config``,
    nothing else -- no optimizer, loop or callback state).

Loading is always STRICT: every parameter and buffer must match the model built from the
config, so a config/checkpoint mismatch fails loudly instead of leaving layers at random init.
"""
import hashlib

import torch
from omegaconf import DictConfig, OmegaConf

FORMAT_VERSION = 1


def state_dict_sha256(sd) -> str:
    """Order-independent content hash of a state_dict (names, dtypes, shapes and values)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def read_checkpoint(path):
    """(state_dict, embedded config or None) from either supported format."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(f"{path}: not a DiG2 checkpoint (no state_dict)")
    cfg = ckpt.get("dig2_config")
    if cfg is not None:
        cfg = OmegaConf.create(cfg)
    return ckpt["state_dict"], cfg


def load_weights_strict(model, path) -> int:
    """Weights-only strict load into an already-built model. Returns the number of tensors.

    Weights-only (not `trainer.fit(ckpt_path=...)`): restoring Lightning's loop state from an
    epoch-boundary checkpoint makes it skip the first validation, and a fresh optimizer is what
    fine-tuning from a released model needs anyway.
    """
    sd, _ = read_checkpoint(path)
    model.load_state_dict(sd, strict=True)
    return len(sd)


# Config keys that determine the parameter shapes or the domain contract of a trained model.
# A release checkpoint embeds exactly these, and loading re-applies them over the run config.
ARCH_KEYS = (
    "model",
    "general.conditional",
    "general.conditional_domain",
    "general.domain_embedding",
    "general.domain_emb_dim",
    "general.cfg_null_mode",
    "general.structural_encoder",
    "general.progressive_readout",
    "dataset.name",
    "dataset.remove_h",
    "dataset.unified.domain_order",
    "dataset.unified.schema_version",
)


def architecture_config(cfg: DictConfig) -> dict:
    """The ARCH_KEYS subset of a full config, as a plain nested dict."""
    out = OmegaConf.create({})
    for key in ARCH_KEYS:
        val = OmegaConf.select(cfg, key)
        if val is not None:
            OmegaConf.update(out, key, val, force_add=True)
    return OmegaConf.to_container(out, resolve=True)


def override_keys(overrides) -> list:
    """Dotted keys set by Hydra command-line overrides (`a.b=1`, `+a.b=1`, `++a.b=1`)."""
    keys = []
    for o in overrides or []:
        key = str(o).split("=", 1)[0].lstrip("+~")
        if key and not key.startswith("hydra") and "@" not in key and key != "experiment":
            keys.append(key)
    return keys


def merge_embedded_config(cfg: DictConfig, path, keep=()) -> DictConfig:
    """Overlay the architecture recorded in a release checkpoint onto `cfg`, so the model is
    always rebuilt with the exact settings its weights were trained with.

    Keys in `keep` (the ones the user set explicitly, see `override_keys`) are restored after the
    overlay, so a command-line option such as `model.prior_mode=default` is not silently undone.
    An explicit override that does change the architecture still fails at the strict load."""
    _, embedded = read_checkpoint(path)
    if embedded is None:
        return cfg
    struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    kept = {k: OmegaConf.select(cfg, k) for k in keep}
    for key in ARCH_KEYS:
        val = OmegaConf.select(embedded, key)
        if val is not None:
            OmegaConf.update(cfg, key, val, merge=False, force_add=True)
    for key, val in kept.items():
        OmegaConf.update(cfg, key, val, merge=False, force_add=True)
    OmegaConf.set_struct(cfg, struct)
    return cfg
