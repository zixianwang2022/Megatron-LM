# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# >>>
import os
# from typing import List

# from megatron.core.models.retro.data.config import RetroPreprocessingConfig
# from megatron.core.models.retro.data.utils import get_gpt_data_dir

# from .multi_split_gpt_dataset import MultiSplitGPTDatasetConfig
# <<<


def get_query_dir(project_dir):
    return os.path.join(project_dir, "query")


def get_neighbor_dir(project_dir, key, dataset):
    return os.path.join(get_query_dir(project_dir), os.path.basename(f"{key}_{dataset.unique_description_hash}"))
