# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

'''Container dataclass for GPT chunk datasets (train, valid, and test).'''

from dataclasses import dataclass

from megatron.core.models.retro.data.query.gpt_chunk_dataset import GPTChunkDataset


@dataclass
class RetroGPTChunkDatasets:
    '''Container dataclass for GPT chunk datasets.'''

    train: GPTChunkDataset = None
    valid: GPTChunkDataset = None
    test: GPTChunkDataset = None
