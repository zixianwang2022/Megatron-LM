# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

? ? ?

from dataclasses import dataclass

from .bert_embedders import RetroBertEmbedders
from .config import RetroPreprocessingConfig
from .gpt_datasets import RetroGPTDatasets
from .tokenizers import RetroTokenizers


@dataclass
class RetroPreprocessingEnv:

    config: RetroPreprocessingConfig
    bert_embedders: RetroBertEmbedders
    gpt_datasets: RetroGPTDatasets
    tokenizers: RetroTokenizers
