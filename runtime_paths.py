"""Filesystem and DataLoader options for the public distribution.

This module uses only the Python standard library. Dataset protocols are
defined in the original dataset modules and are not changed here.
"""
import os
from pathlib import Path


def data_path(*parts):
    return str(Path(os.environ.get("TIGSIGNET_DATA_ROOT", "data")).joinpath(*parts))


def runtime_batch_size(default):
    return int(os.environ.get("TIGSIGNET_BATCH_SIZE", default))


def runtime_workers(default=8):
    return int(os.environ.get("TIGSIGNET_WORKERS", default))


def configure_runtime(args):
    os.environ["TIGSIGNET_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())
    os.environ["TIGSIGNET_BATCH_SIZE"] = str(args.batch_size)
    os.environ["TIGSIGNET_WORKERS"] = str(args.workers)
