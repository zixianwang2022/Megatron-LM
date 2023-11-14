# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import abc
import os
import torch

# >>>
# from megatron import get_retro_args
# from megatron.core.models.retro.data.external_libs import faiss

from .utils import get_index_dir
# <<<


class Index(abc.ABC):

    '''Abstract base class for indexes.

    *Note* : While currently only Faiss-based classes are implemented, in the
    future, this class will be extended with other types of indexes that have
    different performance-accuracy trade-offs.

    The primary methods to override are:
    - train() : Train index on the sampled training chunks.
    - add() : Add all training chunks to index.
    '''

    @classmethod
    def c_verbose(cls, index, v):
        '''Make index object verbose.'''
        assert isinstance(v, bool)
        faiss.ParameterSpace().set_index_parameter(index, "verbose", v)

    def get_empty_index_path(self, env):
        return os.path.join(
            get_index_dir(env),
            "empty_%.3f.faissindex" % env.config.retro_index_train_load_fraction,
        )

    def get_empty_index(self, env):
        return faiss.read_index(self.get_empty_index_path(env))

    def get_added_index_path(self, env):
        return os.path.join(
            get_index_dir(env),
            "added_%.3f_%.3f.faissindex" % (
                env.config.retro_index_train_load_fraction,
                env.config.retro_index_add_load_fraction,
            ),
        )

    def get_added_index(self, env):
        return faiss.read_index(self.get_added_index_path(env))

    @abc.abstractmethod
    def train(self, *args):
        pass

    @abc.abstractmethod
    def add(self, *args):
        pass

    def embed_text_dataset_block(self, embedder, text_dataset, _range):
        '''Embed a range of a text dataset.'''
        sub_dataset = torch.utils.data.Subset(text_dataset, range(*_range))
        return embedder.embed_text_dataset(sub_dataset)
