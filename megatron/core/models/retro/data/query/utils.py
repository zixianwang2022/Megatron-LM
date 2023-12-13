# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
from typing import List

from megatron.core.models.retro.data.config import RetroPreprocessingConfig
from megatron.core.models.retro.data.utils import get_gpt_data_dir

from .multi_split_gpt_dataset import MultiSplitGPTDatasetConfig


def get_query_dir(project_dir):
    return os.path.join(project_dir, "query")


def get_neighbor_dir(project_dir, key, dataset):
    return os.path.join(get_query_dir(project_dir), os.path.basename(f"{key}_{dataset.unique_description_hash}"))


# >>>
# def core_retro_dataset_config_from_args(args, retro_args):
#     return MultiSplitGPTDatasetConfig(
#         is_built_on_rank=is_dataset_built_on_rank,
#         random_seed=retro_args.retro_gpt_seed,
#         sequence_length=retro_args.retro_gpt_seq_length,
#         blend=args.data_path if args.data_path is not None else retro_args.retro_gpt_data_path,
#         split=args.split,
#         path_to_cache=args.data_cache_path,
#         return_document_ids=retro_args.retro_return_doc_ids,
#         split_preprocessing=retro_args.retro_gpt_split,
#     )
# +++
# def core_gpt_dataset_config_from_retro_preprocessing_config(
#     config: RetroPreprocessingConfig,
#     is_dataset_built_on_rank: bool,
# ) -> GPTDatasetConfig:
#     data_dir = get_gpt_data_dir(config.retro_project_dir)
#     blend = list(config.retro_gpt_data_path)
#     for i in range(len(blend) - 1, -1, -2):
#         blend[i] = os.path.join(data_dir, blend[i])
#     return GPTDatasetConfig(
#         is_built_on_rank=is_dataset_built_on_rank,
#         random_seed=config.retro_gpt_seed,
#         sequence_length=config.retro_gpt_seq_length,
#         blend=blend,
#         split=config.retro_gpt_split,
#         path_to_cache=config.retro_gpt_data_cache_path,
#         return_document_ids=True,
#     )
# +++
def core_multi_split_gpt_dataset_config_from_retro_preprocessing_config(
    config: RetroPreprocessingConfig,
    split: str,
    return_document_ids: bool,
    is_dataset_built_on_rank: bool,
    custom_data_path: List[str] = None,
) -> MultiSplitGPTDatasetConfig:
    data_dir = get_gpt_data_dir(config.retro_project_dir)
    if custom_data_path is not None:
        blend=custom_data_path
    else:
        blend = list(config.retro_gpt_data_path)
        for i in range(len(blend) - 1, -1, -2):
            blend[i] = os.path.join(data_dir, blend[i])
    # >>>
    if not return_document_ids:
        raise Exception("return_document_ids? %s" % return_document_ids)
    # <<<
    config = MultiSplitGPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=config.retro_gpt_seed,
        sequence_length=config.retro_gpt_seq_length,
        blend=blend,
        split=split,
        split_preprocessing=config.retro_gpt_split,
        path_to_cache=config.retro_gpt_data_cache_path,
        return_document_ids=return_document_ids,
    )
    # >>>
    if not return_document_ids or custom_data_path is not None:
        from lutil import pax
        pax("config")
    # <<<
    return config
# <<<
