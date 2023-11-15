# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os


def get_query_dir(env):
    return os.path.join(env.config.retro_project_dir, "query")


def get_neighbor_dirname(env, key, dataset):
    return os.path.join(get_query_dir(env), os.path.basename(f"{key}_{dataset.unique_description_hash}"))
