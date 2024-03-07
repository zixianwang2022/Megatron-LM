# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

'''Container dataclass for holding both in-memory and on-disk Bert embedders.'''

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


class Embedder(abc.ABC):
    '''Base class for all Bert embedders.

    All embedders should be able to embed either an entire text dataset (to a 2D
    numpy array), or a single text string (to a 1D numpy array).
    '''

    @abc.abstractmethod
    def embed_text_dataset(self, text_dataset: torch.utils.data.Dataset) -> np.ndarray:
        '''Embed a text dataset.'''

    @abc.abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        '''Embed a simple string of text.'''


@dataclass
class RetroBertEmbedders:
    '''Container dataclass for in-memory and on-disk Bert embedders.'''

    disk: Embedder
    mem: Embedder
