# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import os


def get_config_path(project_dir: str) -> str:
    '''Config copy stored within retro project dir.'''
    return os.path.join(project_dir, "config.json")


def get_gpt_data_dir(project_dir: str) -> str:
    '''Get project-relative directory of GPT bin/idx datasets.'''
    return os.path.join(project_dir, "data")
