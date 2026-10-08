"""Write a release checkpoint: the model weights plus the architecture config they need.

    python -m dig2.tools.export_ckpt SOURCE.ckpt OUT.ckpt [experiment overrides ...]

SOURCE is any checkpoint whose state_dict fits the DiG2 architecture (e.g. a Lightning
checkpoint from `dig2.train`). Everything except the weights is dropped: optimizer, scheduler,
loop and callback state, and the training run's hyperparameters. The architecture recorded in
OUT is DiG2's base config with the given overrides (e.g. `+experiment=readout_3_6_9_12`); the export
verifies that the weights load strictly into a model built from it.

OUT contains exactly: state_dict, dig2_config, format_version, state_dict_sha256.
"""
import argparse
import os

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dig2 import paths
from dig2.checkpoint import FORMAT_VERSION, architecture_config, state_dict_sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. +experiment=readout_3_6_9_12")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the strict-load check (it builds the datamodule, which needs the data)")
    a = ap.parse_args()

    src = torch.load(a.source, map_location="cpu", weights_only=False)
    sd = {k: v.detach().cpu() for k, v in src["state_dict"].items()}
    with initialize_config_dir(config_dir=str(paths.repo_root() / "dig2" / "configs"), version_base="1.3"):
        cfg = compose(config_name="dig2", overrides=list(a.overrides))

    if not a.no_verify:
        from dig2.pipeline import build_model
        OmegaConf.set_struct(cfg, False)
        cfg.hier_dir = os.path.dirname(os.path.abspath(a.out))
        model, _ = build_model(cfg, compute_reference=False)
        model.load_state_dict(sd, strict=True)
        print(f"[export] strict load OK: {len(sd)} tensors")

    out = {
        "state_dict": sd,
        "dig2_config": architecture_config(cfg),
        "format_version": FORMAT_VERSION,
        "state_dict_sha256": state_dict_sha256(sd),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    print(f"[export] wrote {a.out}  sha256(state_dict)={out['state_dict_sha256']}")


if __name__ == "__main__":
    main()
