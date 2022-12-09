# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import faiss
import numpy as np
import os
import torch

from tools.retro import utils


class Index:

    @classmethod
    def c_verbose(cls, index, v):
        '''Make index object verbose.'''
        assert isinstance(v, bool)
        faiss.ParameterSpace().set_index_parameter(index, "verbose", v)
        # index.verbose = True # ... maybe?


    @classmethod
    def swig_ptr(cls, x):
        '''Get raw C++ pointer.'''
        return faiss.swig_ptr(np.ascontiguousarray(x))


    @classmethod
    def get_empty_index_path(cls, dir_path):
        return os.path.join(dir_path, "empty.faissindex")


    @classmethod
    def get_added_index_path(cls, input_data_paths, dir_path):
        return os.path.join(dir_path, "added.faissindex")


    # @classmethod
    # def get_output_data_path(cls, dir_path, task, suffix):
    #     return os.path.join(dir_path, "%s_output%s_%s.hdf5" % (task, suffix))


    # @classmethod
    # def get_output_data_path(cls, dir_path, task, suffix):
    #     sub_dir_name = "%s_output" % task
    #     utils.make_sub_dir(dir_path, sub_dir_name)
    #     return os.path.join(dir_path, sub_dir_name, "%s.hdf5" % suffix)


    # def get_missing_output_data_path_map(self, input_paths, dir_path, task):

    #     all_output_paths = []
    #     missing_output_path_map = {}
    #     missing_index = 0
    #     for input_index, input_path in enumerate(input_paths):
    #         output_path = self.get_output_data_path(dir_path, task, input_index)
    #         all_output_paths.append(output_path)
    #         if not os.path.isfile(output_path):
    #             if missing_index % torch.distributed.get_world_size() == \
    #                torch.distributed.get_rank():
    #                 missing_output_path_map[input_index] = output_path
    #             missing_index += 1

    #     torch.distributed.barrier()

    #     return all_output_paths, missing_output_path_map


    def train(self, *args):
        raise Exception("implement 'train()' for <%s>." % type(self).__name__)


    def add(self, *args):
        raise Exception("implement 'add()' for <%s>." % type(self).__name__)


    def embed_text_dataset_block(self, embedder, text_dataset, _range):
        '''Embed a range of a text dataset.'''
        sub_dataset = torch.utils.data.Subset(text_dataset, range(*_range))
        return embedder.embed_text_dataset(sub_dataset, len(text_dataset))
