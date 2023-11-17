# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os


def get_query_dir(config):
    return os.path.join(config.retro_project_dir, "query")


def get_neighbor_dir(config, key, dataset):
    # >>>
    # from lutil import pax
    # pax({
    #     "top hash" : dataset.unique_description_hash,
    #     "sub hashes" : [ d.unique_description_hash for d in dataset.datasets ],
    # })
    # <<<
    return os.path.join(get_query_dir(config), os.path.basename(f"{key}_{dataset.unique_description_hash}"))
