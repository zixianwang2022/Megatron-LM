# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import os
from importlib.metadata import version

import torch
from pkg_resources import packaging


def get_config_path(project_dir: str) -> str:
    """Config copy stored within retro project dir."""
    return os.path.join(project_dir, "config.json")


def get_gpt_data_dir(project_dir: str) -> str:
    """Get project-relative directory of GPT bin/idx datasets."""
    return os.path.join(project_dir, "data")


def get_dummy_mask(size, device):
    te_version = packaging.version.Version(version("transformer-engine"))
    if te_version >= packaging.version.Version("1.3"):
        return torch.full(size=size, fill_value=True, dtype=torch.bool, device=device)
    else:
        return None
