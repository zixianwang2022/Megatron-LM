# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from .indexes import FaissBaseIndex, FaissParallelAddIndex


class IndexFactory:
    '''Get index.

    Index type generally read from argument '--retro-index-ty'.
    '''

    @classmethod
    def get_index_ty(cls, index_ty):
        return {
            "faiss-base" : FaissBaseIndex,
            "faiss-par-add" : FaissParallelAddIndex,
        }[index_ty]


    @classmethod
    def get_index(cls, index_ty):
        index_ty = cls.get_index_ty(index_ty)
        index = index_ty()
        return index
