"""Release checkpoints carry the weights and their architecture, and nothing else."""
import os
import pathlib
import pickle
import re

import pytest
import torch

from dig2.checkpoint import ARCH_KEYS, FORMAT_VERSION, state_dict_sha256

from _build import ROOT

CKPT_DIR = pathlib.Path(os.environ.get("DIG2_TEST_CKPT_DIR", str(ROOT / "checkpoints")))
CKPTS = sorted(CKPT_DIR.glob("*.ckpt"))
TOP_LEVEL = {"state_dict", "dig2_config", "format_version", "state_dict_sha256"}


@pytest.mark.ckpt
@pytest.mark.parametrize("path", CKPTS, ids=[p.name for p in CKPTS])
def test_release_format(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ckpt) == TOP_LEVEL
    assert ckpt["format_version"] == FORMAT_VERSION
    assert ckpt["state_dict_sha256"] == state_dict_sha256(ckpt["state_dict"])
    top = {k.split(".")[0] for k in ARCH_KEYS}
    assert set(ckpt["dig2_config"]) <= top


@pytest.mark.ckpt
@pytest.mark.parametrize("path", CKPTS, ids=[p.name for p in CKPTS])
def test_release_checkpoint_has_only_architecture_metadata(path):
    """The non-tensor part of a release checkpoint is the architecture config alone."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = pickle.dumps({k: v for k, v in ckpt.items() if k != "state_dict"})
    meta += pickle.dumps(sorted(ckpt["state_dict"]))
    assert not re.search(rb"(?i)optimizer|lr_scheduler|callbacks|hyper_parameters|loops", meta)


@pytest.mark.ckpt
def test_explicit_override_survives_embedded_config():
    """The embedded architecture is overlaid on the defaults, but a key the user set on the
    command line wins; without `keep` the same key reverts to the checkpoint's value."""
    from dig2.checkpoint import merge_embedded_config, override_keys
    from omegaconf import OmegaConf

    from _build import compose_cfg

    path = CKPT_DIR / "dig2_base.ckpt"
    if not path.exists():
        pytest.skip(f"{path} not present")
    embedded = torch.load(path, map_location="cpu", weights_only=False)["dig2_config"]
    shipped = embedded["model"]["prior_mode"]
    other = "default" if shipped != "default" else "domain_specific"
    overrides = [f"model.prior_mode={other}"]
    assert override_keys(["+experiment=eval", f"++model.prior_mode={other}", "hydra.run.dir=x"]) \
        == ["model.prior_mode"]

    kept = merge_embedded_config(compose_cfg(overrides), path, keep=override_keys(overrides))
    assert kept.model.prior_mode == other
    dropped = merge_embedded_config(compose_cfg(overrides), path)
    assert dropped.model.prior_mode == shipped
    assert OmegaConf.select(kept, "dataset.unified.domain_order") == \
        list(embedded["dataset"]["unified"]["domain_order"])
