import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CKPT = pathlib.Path(os.environ.get("DIG2_TEST_CKPT_DIR", str(ROOT / "checkpoints")))


def pytest_collection_modifyitems(config, items):
    """Skip tests whose prerequisites are missing instead of failing them."""
    import torch
    have_gpu = torch.cuda.is_available()
    datadir = os.environ.get("DIG2_TEST_DATADIR", str(ROOT / "data"))
    have_data = (pathlib.Path(datadir) / "unified").exists()
    for item in items:
        if "gpu" in item.keywords and not have_gpu:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if "data" in item.keywords and not have_data:
            item.add_marker(pytest.mark.skip(reason=f"no unified dataset cache under {datadir}"))
        if "ckpt" in item.keywords and not any(CKPT.glob("*.ckpt")):
            item.add_marker(pytest.mark.skip(reason="no checkpoints in checkpoints/"))


@pytest.fixture(autouse=True, scope="session")
def _reuse_unified_datamodule():
    """Loading the unified RAM cache takes minutes; build it once per dataset config and share it
    across tests. Its contents depend only on `cfg.dataset` and `general.conditional_domain`, so
    tests that vary model or sampling settings still get the datamodule they would have built."""
    from omegaconf import OmegaConf

    import dig2.data.unified.datamodule as dm_mod

    real = dm_mod.UnifiedDataModule
    built = {}

    def cached(cfg):
        key = (OmegaConf.to_yaml(cfg.dataset), bool(getattr(cfg.general, "conditional_domain", False)))
        if key not in built:
            built[key] = real(cfg)
        dm = built[key]
        dm.cfg = cfg
        return dm

    dm_mod.UnifiedDataModule = cached
    yield
    dm_mod.UnifiedDataModule = real
