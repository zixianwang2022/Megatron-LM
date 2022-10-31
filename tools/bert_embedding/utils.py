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
import torch
from tqdm import tqdm

from megatron import mpu, print_rank_0

# >>>
from lutil import pax
# <<<


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


# def get_missing_embedding_blocks(workdir, dataset, block_size):
#     n_samples = len(dataset)
def get_missing_blocks_by_rank(workdir, n_samples, block_size,
                               validate = lambda f : None):

    # Block ranges.
    block_start_idxs = list(range(0, n_samples, block_size))
    block_end_idxs = [ min(n_samples, i + block_size) for i in block_start_idxs ]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # All block files (existing + missing).
    n_digits = int(np.ceil(np.log(n_samples) / np.log(10)) + 1)
    all_block_items = [{
        "range" : r,
        "path" : os.path.join(
            workdir,
            "%s-%s.hdf5" % tuple([ str(i).zfill(n_digits) for i in r ]),
        )
    } for r in block_ranges]

    # Delete corrupt files.
    if torch.distributed.get_rank() == 0:
        existing_block_paths = [item["path"]
                                for item in all_block_items
                                if os.path.exists(item["path"])]
        pbar = tqdm(existing_block_paths)
        for index, path in enumerate(pbar):
            pbar.set_description("validating block.")

            try:
                f = h5py.File(path, "r")
            except:
                raise Exception("unable to open/validate '%s'." % path)

            try:
                # assert f["data"].shape[1] == 1024
                validate(f)
            except:
                raise Exception("delete block file.")
                os.remove(path)
            finally:
                f.close()

    # Wait for files to be deleted.
    torch.distributed.barrier()

    # Filter missing files.
    missing_block_items = [item
                           for item in all_block_items
                           if not os.path.exists(item["path"])]

    # This rank's missing files.
    data_parallel_rank = mpu.get_data_parallel_rank()
    data_parallel_world_size = mpu.get_data_parallel_world_size()
    rank_missing_block_items = missing_block_items[data_parallel_rank:len(missing_block_items):data_parallel_world_size]

    # Extend rank's missing items (with None) such that all ranks have equal
    # length lists. This allows for easier tracking of global progress.
    n_missing_tensor = torch.cuda.LongTensor([len(rank_missing_block_items)])
    torch.distributed.all_reduce(n_missing_tensor,
                                 op = torch.distributed.ReduceOp.MAX)
    max_n_missing = n_missing_tensor.item()
    rank_missing_block_items += \
        [None] * (max_n_missing - len(rank_missing_block_items))

    # >>>
    # print_seq("missing blocks [%d] : %s ... %s." % (
    #     len(rank_missing_block_items),
    #     str(rank_missing_block_items[0]["range"]),
    #     str(rank_missing_block_items[-1]["range"]) if rank_missing_block_items[-1] else str(rank_missing_block_items[-2]["range"]),
    # ))
    # <<<

    return len(missing_block_items), rank_missing_block_items
