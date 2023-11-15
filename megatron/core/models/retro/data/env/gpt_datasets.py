# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass

from megatron.core.datasets.gpt_dataset import GPTDataset


@dataclass
class RetroGPTDatasets:

    train: GPTDataset = None
    valid: GPTDataset = None
    test: GPTDataset = None
