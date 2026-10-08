"""Shared helpers: compose a DiG2 config the way `dig2.train` does and build the model."""
import os
import pathlib

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]


def compose_cfg(overrides=()):
    datadir = os.environ.get("DIG2_TEST_DATADIR")
    extra = [f"dataset.datadir={datadir}"] if datadir else []
    with initialize_config_dir(config_dir=str(ROOT / "dig2" / "configs"), version_base="1.3"):
        cfg = compose(config_name="dig2", overrides=extra + list(overrides))
    OmegaConf.set_struct(cfg, False)
    cfg.hier_dir = str(ROOT / "outputs" / "tests")
    os.makedirs(cfg.hier_dir, exist_ok=True)
    return cfg
