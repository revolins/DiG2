"""Domain interpolation is exact at its endpoints and connected in between.

alpha = 1 must reproduce plain ZINC conditioning bitwise, alpha = 0 must hand the model exactly
QM9's conditioning rows, and alpha = 1e-6 must move them. Under a single global noise prior
(`model.prior_mode=default`) alpha = 0 must also reproduce plain QM9 outputs bitwise; under the
shipped domain-specific prior it is not required to (the prior stays at the destination domain).
"""
import os
import pathlib

import pytest

from _build import ROOT

CKPT_DIR = pathlib.Path(os.environ.get("DIG2_TEST_CKPT_DIR", str(ROOT / "checkpoints")))
CKPT = CKPT_DIR / "dig2_base.ckpt"


@pytest.mark.ckpt
@pytest.mark.data
@pytest.mark.parametrize("prior_mode", ["domain_specific", "default"])
def test_interp_endpoints(prior_mode, tmp_path):
    if not CKPT.exists():
        pytest.skip(f"{CKPT} not present")
    from dig2.tools.domain_interp import endpoint_check
    from dig2.tools.domain_interp_common import build_and_load

    overrides = [f"model.prior_mode={prior_mode}", "sample.sample_steps=10"]
    datadir = os.environ.get("DIG2_TEST_DATADIR")
    if datadir:
        overrides.append(f"dataset.datadir={datadir}")
    model, dm, _, device, _ = build_and_load(str(CKPT), overrides=overrides,
                                             scratch_dir=str(tmp_path), device="cpu")
    res = endpoint_check(model, dm, device, batch_size=4, num_nodes=14, seed=0,
                         src="qm9", dst="zinc")

    assert res["alpha0_rows_equal_src_rows"]
    assert res["alpha1_rows_equal_dst_rows"]
    assert res["alpha1_output_equals_plain_dst"]
    assert res["epsilon_rows_differ_from_dst_rows"]
    if prior_mode == "default":
        assert res["alpha0_output_equality_required"]
        assert res["alpha0_output_equals_plain_src"]
    assert res["passed"]
