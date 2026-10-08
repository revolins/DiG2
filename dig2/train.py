"""DiG2 training / evaluation entry point.

    python -m dig2.train                                   # train the multi-domain model
    python -m dig2.train +experiment=eval                  # evaluate checkpoints/dig2.ckpt
    python -m dig2.train +experiment=anchor_ft general.eval_resume=<base ckpt>

See README.md for every procedure.
"""
import os
import warnings

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from dig2.checkpoint import load_weights_strict, merge_embedded_config, override_keys
from dig2.ddp_utils import configured_world_size, ddp_preflight
from dig2.pipeline import build_model

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def _devices(cfg):
    gpus = cfg.general.gpus
    if isinstance(gpus, (list, ListConfig)):
        return [int(g) for g in gpus]
    return gpus


def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    if getattr(cfg, "hier_dir", None) is None:
        cfg.hier_dir = HydraConfig.get().runtime.output_dir
    # Relative checkpoint paths are relative to where the command was launched (Hydra changes
    # into the run directory).
    eval_resume = getattr(cfg.general, "eval_resume", None)
    if eval_resume:
        eval_resume = cfg.general.eval_resume = to_absolute_path(str(eval_resume))
    if cfg.general.test_only and cfg.general.test_only is not True:
        cfg.general.test_only = to_absolute_path(str(cfg.general.test_only))
    if eval_resume:
        if cfg.general.resume:
            raise ValueError(
                "general.eval_resume (weights only, fresh loop state) and general.resume "
                "(restores loop state) are mutually exclusive."
            )
        cfg = merge_embedded_config(cfg, eval_resume,
                                    keep=override_keys(HydraConfig.get().overrides.task))

    world_size = configured_world_size(cfg)
    ddp_preflight(cfg)
    devices = _devices(cfg)
    use_gpu = torch.cuda.is_available() and devices not in (None, [], 0)

    pl.seed_everything(cfg.train.seed)
    if use_gpu and world_size > 1:
        # Pin THIS rank to its own card before anything touches CUDA, otherwise every rank also
        # opens a context on cuda:0 and rank 0 runs out of memory first.
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(devices[rank] if rank < len(devices) else devices[0])
    elif use_gpu and isinstance(devices, list):
        torch.cuda.set_device(devices[0])

    model, datamodule = build_model(cfg)

    callbacks = []
    if cfg.train.save_model:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(str(cfg.hier_dir), "checkpoints"),
            filename="{epoch}",
            save_top_k=-1,
            every_n_epochs=int(
                getattr(cfg.general, "ckpt_every_n_epochs", None)
                or (cfg.general.sample_every_val * cfg.general.check_val_every_n_epochs)
            ),
            # Fire at train-epoch end, so checkpoint cadence is decoupled from validation.
            save_on_train_epoch_end=True,
        ))

    trainer_extra = OmegaConf.to_container(cfg.trainer, resolve=True) if "trainer" in cfg and cfg.trainer else {}
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true" if world_size > 1 else "auto",
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices if use_gpu else 1,
        use_distributed_sampler=False,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=bool(getattr(cfg.general, "fast_dev_run", False)),
        enable_progress_bar=bool(getattr(cfg.train, "progress_bar", False)),
        callbacks=callbacks,
        log_every_n_steps=50,
        logger=CSVLogger(save_dir=str(cfg.hier_dir), name="lightning_logs"),
        **trainer_extra,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dig2] parameters: {n_params:,} (trainable "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,})", flush=True)

    if eval_resume:
        n = load_weights_strict(model, eval_resume)
        print(f"[dig2] eval_resume: strict load of {n} tensors from {eval_resume}", flush=True)

    test_only = cfg.general.test_only
    if test_only:
        # `true`: test the weights already loaded through eval_resume (release checkpoints);
        # a path: a Lightning checkpoint written by this script.
        trainer.test(model, datamodule=datamodule,
                     ckpt_path=None if test_only is True else str(test_only))
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)


@hydra.main(version_base="1.3", config_path="configs", config_name="dig2")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
