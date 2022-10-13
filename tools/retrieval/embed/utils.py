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

from collections import defaultdict
import glob
import h5py
import numpy as np
import os

from megatron import print_rank_0
from tools.retrieval.chunks.utils import get_chunk_index_path_map

# >>>
from lutil import pax
# <<<


def get_embedding_path_map(workdir):

    # Directory map.
    chunk_index_path_map = get_chunk_index_path_map(workdir)
    embedding_path_map = {}
    for key, chunk_index_path in chunk_index_path_map.items():

        # Embedding sub-directory.
        embedding_dir = os.path.join(workdir, "embed", key)
        os.makedirs(embedding_dir, exist_ok = True)

        # Sort data paths for reproducibility.
        embedding_path_map[key] = {
            "dir" : embedding_dir,
            "data" : sorted(glob.glob(embedding_dir + "/*.hdf5")),
        }

    # pax(0, {
    #     "chunk_index_path_map" : chunk_index_path_map,
    #     "embedding_path_map" : embedding_path_map,
    #     "workdir" : workdir,
    # })

    return embedding_path_map


def save_data(data_map, *args):

    if len(args) == 1:
        path = args[0]
    elif len(args) == 2:
        dir_path, file_name = args
        path = os.path.join(dir_path, file_name)
    else:
        raise Exception("specialize for len(args) == %d." % len(args))

    if not os.path.isfile(path):
        f = h5py.File(path, "w")
        # f.create_dataset("data", data = input_data)
        for k, v in data_map.items():
            f.create_dataset(k, data = v)
        f.close()

    return path


def load_data(paths, timer):

    timer.push("shape")
    shape_map = defaultdict(lambda : (0, None))
    for p in paths:
        f = h5py.File(p, "r")
        for k in f.keys():
            shape = tuple(f[k].shape)
            shape_map[k] = (shape_map[k][0] + shape[0], shape[1])
        f.close()
    timer.pop()

    timer.push("alloc")
    data_map = { k : np.empty(s, dtype = "f4") for k, s in shape_map.items() }
    start_map = { k : 0 for k in shape_map }
    timer.pop()

    timer.push("load")
    for pi, p in enumerate(paths):
        print_rank_0("load path %d / %d ... '%s'." % (pi, len(paths), p))
        f = h5py.File(p, "r")
        for k in f.keys():
            i0 = start_map[k]
            i1 = i0 + len(f[k])
            data_map[k][i0:i1] = f[k]
            start_map[k] += len(f[k])
        f.close()
    timer.pop()

    # pax(0, {
    #     "paths" : paths,
    #     "shape_map" : shape_map,
    #     "start_map" : start_map,
    #     "data_map" : data_map,
    # })
    
    return data_map

