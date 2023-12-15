# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass

# >>>
# from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.models.retro.data.query.gpt_chunk_dataset import GPTChunkDataset
# <<<


# >>>
# @dataclass
# class RetroGPTDatasets:

#     train: GPTDataset = None
#     valid: GPTDataset = None
#     test: GPTDataset = None
# +++
@dataclass
class RetroGPTChunkDatasets:

    train: GPTChunkDataset = None
    valid: GPTChunkDataset = None
    test: GPTChunkDataset = None
# <<<
