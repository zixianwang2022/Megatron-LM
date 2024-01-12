# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os

from megatron.core.datasets.megatron_dataset import MegatronDataset


def get_query_dir(project_dir: str) -> str:
    return os.path.join(project_dir, "query")


def get_neighbor_dir(project_dir: str, key: str, dataset: MegatronDataset) -> str:
    return os.path.join(
        get_query_dir(project_dir), os.path.basename(f"{key}_{dataset.unique_description_hash}"),
    )
