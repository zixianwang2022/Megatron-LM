# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""The IndexFactory constructs an index from an index type string."""

from megatron.core.models.retro.data.index.index import Index

from .indexes import FaissBaseIndex, FaissParallelAddIndex


class IndexFactory:
    """Get index.

    Index type generally read from argument '--retro-index-ty'.
    """

    @classmethod
    def get_index_class(cls, index_type: str) -> type:
        """Get an index class, given a type string."""
        return {"faiss-base": FaissBaseIndex, "faiss-par-add": FaissParallelAddIndex,}[index_type]

    @classmethod
    def get_index(cls, index_type: str) -> Index:
        """Construct an index from an index type string."""
        index_class = cls.get_index_class(index_type)
        index = index_class()
        return index
