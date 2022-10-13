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

import os
import shutil
import torch

from tools.retrieval.embed.utils import get_embedding_path_map
from tools.retrieval.index.factory import IndexFactory

from .utils import get_index_workdir

# >>>
from lutil import pax
# <<<


# def get_workdir(args):
#     workdir = os.path.join(args.retrieval_workdir, "index")
#     os.makedirs(workdir, exist_ok = True)
#     return workdir


def train_index(args, timer):
    # workdir = get_workdir(args)
    workdir = get_index_workdir(args)
    embedding_path_map = get_embedding_path_map(args.retrieval_workdir)
    input_data_paths = embedding_path_map["sampled"]["data"]
    index = IndexFactory.get_index(args)
    index.train(input_data_paths, workdir, timer)


def add_to_index(args, timer):
    workdir = get_index_workdir(args)
    embedding_path_map = get_embedding_path_map(args.retrieval_workdir)
    input_data_paths = embedding_path_map["full"]["data"]
    index = IndexFactory.get_index(args)
    output_index_path = index.add(input_data_paths, workdir, timer)
    # pax(0, {
    #     "input_data_paths" : input_data_paths,
    #     "output_index_path" : output_index_path,
    #     "index" : index,
    # })
    return output_index_path


def remove_add_outputs(args, timer):

    # Single process only.
    if torch.distributed.get_rank() != 0:
        return

    # Get file paths.
    add_paths = [
        os.path.join(args.index_dir_path, r, n)
        for r, ds, fs in os.walk(args.index_dir_path)
        for n in [ *ds, *fs ]
        if n.startswith("add")
    ]

    # Remove files.
    for p in add_paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)
        else:
            raise Exception("specialize for this monster, '%s'." % p)

