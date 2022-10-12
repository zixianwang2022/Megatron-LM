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
import torch

from .get_paths import get_all_data_paths

# >>>
from lutil import pax
# <<<

def get_nan_stats(args, timer):

    if torch.distributed.get_rank() != 0:
        return

    data_paths = get_all_data_paths(args)

    max_num_paths = 10 # 2
    row_start = 0
    row_count_map = defaultdict(lambda : 0)
    for data_path_index, data_path in enumerate(data_paths):

        if data_path_index == max_num_paths:
            break


        
        print("data path %d / %d." % (data_path_index, len(data_paths)))

        f = h5py.File(data_path, "r")
        # data = np.copy(f["data"])
        # nan_indexes = np.argwhere(np.isnan(data))
        num_rows = len(f["data"])
        nan_indexes = np.argwhere(np.isnan(f["data"]))
        f.close()

        for r, c in nan_indexes:
            row_count_map[r + row_start] += 1

        row_start += num_rows # len(data)

    pax({
        "row_count_map" : row_count_map,
        "num nan rows" : "%d / %d" % (len(row_count_map), row_start),
    })
    pax({"data_paths": data_paths})
