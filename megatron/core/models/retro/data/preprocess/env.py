# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass
from typing import Any

from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig

from .config import RetroPreprocessingConfig


@dataclass
class RetroPreprocessingEnv:

    config: RetroPreprocessingConfig = None
    data_config: GPTDatasetConfig = None
    gpt_tokenizer: Any = None
    bert_tokenizer: Any = None
