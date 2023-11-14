# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import abc
from dataclasses import dataclass
from typing import Any


class Embedder(abc.ABC):
    
    @abc.abstractmethod
    def embed_text_dataset(self, text_dataset):
        pass

    @abc.abstractmethod
    def embed_text(self, text):
        pass

@dataclass
class RetroEmbedders:

    # >>>
    # disk_ty: Callable # = None
    # mem_ty: Callable # = None
    disk: Embedder
    mem: Embedder
    # <<<
