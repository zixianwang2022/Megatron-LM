# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import abc
from dataclasses import dataclass
import numpy as np
import torch
from typing import Any


class Embedder(abc.ABC):

    @abc.abstractmethod
    def embed_text_dataset(self, text_dataset: torch.utils.data.Dataset) -> np.ndarray:
        pass

    @abc.abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        pass


@dataclass
class RetroBertEmbedders:

    disk: Embedder
    mem: Embedder
