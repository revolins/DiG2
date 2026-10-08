"""Filesystem anchors. `dataset.datadir` values in the configs are relative to the repo root
unless given as absolute paths (point them anywhere with e.g. `dataset.datadir=/scratch/data`)."""
import os
import pathlib


def repo_root() -> pathlib.Path:
    return pathlib.Path(os.path.realpath(__file__)).parents[1]


def checkpoints_dir() -> pathlib.Path:
    return repo_root() / "checkpoints"


def default_checkpoint() -> pathlib.Path:
    return checkpoints_dir() / "dig2.ckpt"
