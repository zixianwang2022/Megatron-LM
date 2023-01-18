# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import faiss
import numpy as np
import os
import torch

from .utils import get_index_dir


class Index:

    @classmethod
    def c_verbose(cls, index, v):
        '''Make index object verbose.'''
        assert isinstance(v, bool)
        faiss.ParameterSpace().set_index_parameter(index, "verbose", v)
        # >>>
        # index.verbose = True # ... maybe?
        # <<<


    @classmethod
    def swig_ptr(cls, x):
        '''Get raw C++ pointer.'''
        return faiss.swig_ptr(np.ascontiguousarray(x))


    def get_empty_index_path(self):
        return os.path.join(get_index_dir(), "empty.faissindex")
    def get_empty_index(self):
        return faiss.read_index(self.get_empty_index_path())


    def get_added_index_path(self):
        return os.path.join(get_index_dir(), "added.faissindex")
    def get_added_index(self):
        return faiss.read_index(self.get_added_index_path())


    def train(self, *args):
        raise Exception("implement 'train()' for <%s>." % type(self).__name__)


    def add(self, *args):
        raise Exception("implement 'add()' for <%s>." % type(self).__name__)


    def embed_text_dataset_block(self, embedder, text_dataset, _range):
        '''Embed a range of a text dataset.'''
        sub_dataset = torch.utils.data.Subset(text_dataset, range(*_range))
        return embedder.embed_text_dataset(sub_dataset)
