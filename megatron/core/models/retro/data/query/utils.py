# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os

from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig


def get_query_dir(config):
    return os.path.join(config.retro_project_dir, "query")


def get_data_dir(config):
    return os.path.join(get_query_dir(config), "data")


def get_neighbor_dir(config, key, dataset):
    return os.path.join(get_query_dir(config), os.path.basename(f"{key}_{dataset.unique_description_hash}"))


def core_gpt_dataset_config_from_retro_preprocessing_config(
    config,
    is_dataset_built_on_rank,
    return_document_ids,
):
    data_dir = get_data_dir(config)
    blend = list(config.retro_gpt_data_path)
    for i in range(len(blend) - 1, -1, -2):
        blend[i] = os.path.join(data_dir, blend[i])
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=config.retro_gpt_seed,
        sequence_length=config.retro_gpt_seq_length,
        blend=blend,
        split=config.retro_gpt_split,
        path_to_cache=config.retro_gpt_data_cache_path,
        return_document_ids=return_document_ids,
    )
