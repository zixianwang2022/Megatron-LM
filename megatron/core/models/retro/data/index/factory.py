# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.models.retro.data.index.index import Index

from .indexes import FaissBaseIndex, FaissParallelAddIndex


class IndexFactory:
    '''Get index.

    Index type generally read from argument '--retro-index-ty'.
    '''

    @classmethod
    def get_index_class(cls, index_type: str) -> type:
        return {"faiss-base": FaissBaseIndex, "faiss-par-add": FaissParallelAddIndex,}[index_type]

    @classmethod
    def get_index(cls, index_type: str) -> Index:
        index_class = cls.get_index_class(index_type)
        index = index_class()
        return index
