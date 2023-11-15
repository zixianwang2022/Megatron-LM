# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass

from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig

from .config import RetroPreprocessingConfig
from .embedders import RetroEmbedders
from .tokenizers import RetroTokenizers


@dataclass
class RetroPreprocessingEnv:

    config: RetroPreprocessingConfig # = None
    data_config: GPTDatasetConfig # = None
    embedders: RetroEmbedders # = None
    tokenizers: RetroTokenizers # = None
