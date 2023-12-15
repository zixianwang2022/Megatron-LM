# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass

from megatron.core.models.retro.data.query.gpt_chunk_dataset import GPTChunkDataset


@dataclass
class RetroGPTChunkDatasets:

    train: GPTChunkDataset = None
    valid: GPTChunkDataset = None
    test: GPTChunkDataset = None
