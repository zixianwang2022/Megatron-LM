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
import h5py
import numpy as np
import os

from tools.retrieval.utils import print_rank

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
    
    for pi, p in enumerate(paths):
        print_rank("load path %d / %d ... '%s'." % (pi, len(paths), p))
        timer.push("load")
        f = h5py.File(p, "r")
        for k in f.keys():
            i0 = start_map[k]
            i1 = i0 + len(f[k])
            if 1:
                data_map[k][i0:i1] = f[k]
            else:
                d = np.copy(f[k])
                if np.isnan(d).any():
                    np.nan_to_num(d, copy = False, nan = 0.0)
                data_map[k][i0:i1] = d

                # if np.isnan(f[k]).any():
                #     data_map[k][i0:i1] = np.nan_to_num(f[k], copy = False, nan = 0.0)
                # else:
                #     data_map[k][i0:i1] = f[k]
            start_map[k] += len(f[k])
        f.close()
        timer.pop()

    return data_map
