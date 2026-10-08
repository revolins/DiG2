"""Every release checkpoint strict-loads into the model its embedded config describes."""
import os
import pathlib

import pytest
import torch

from dig2.checkpoint import merge_embedded_config, read_checkpoint, state_dict_sha256
from dig2.pipeline import build_model

from _build import ROOT, compose_cfg

CKPT_DIR = pathlib.Path(os.environ.get("DIG2_TEST_CKPT_DIR", str(ROOT / "checkpoints")))
CKPTS = sorted(CKPT_DIR.glob("*.ckpt"))


@pytest.mark.ckpt
@pytest.mark.data
@pytest.mark.parametrize("path", CKPTS, ids=[p.name for p in CKPTS])
def test_release_checkpoint_strict_load(path):
    sd, embedded = read_checkpoint(path)
    assert embedded is not None, f"{path.name} is not in the release format (no dig2_config)"
    cfg = merge_embedded_config(compose_cfg(), path)
    model, _ = build_model(cfg, compute_reference=False)
    res = model.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    assert state_dict_sha256(model.state_dict()) == state_dict_sha256(sd)
    assert state_dict_sha256(sd) == torch.load(path, weights_only=False)["state_dict_sha256"]


@pytest.mark.ckpt
@pytest.mark.data
def test_wrong_architecture_is_rejected():
    """Negative control: weights trained with readout depths 8/10/12 must NOT load into the
    3/6/9/12 architecture (the gate has a different number of outputs)."""
    path = CKPT_DIR / "dig2_base.ckpt"
    if not path.exists():
        pytest.skip("checkpoints/dig2_base.ckpt not present")
    sd, _ = read_checkpoint(path)
    model, _ = build_model(compose_cfg(["+experiment=readout_3_6_9_12"]), compute_reference=False)
    with pytest.raises(RuntimeError):
        model.load_state_dict(sd, strict=True)
