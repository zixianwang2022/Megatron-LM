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

import glob
import h5py
import numpy as np
import os
import psutil
import shutil
import time
import torch
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from tools.retro.db.utils import get_indexed_dataset_infos

# >>>
from lutil import pax
# <<<


# def get_index_str():
#     """Faiss notation for index structure."""
#     args = get_retro_args()
#     return "OPQ%d_%d,IVF%d_HNSW%d,PQ%d" % (
#         args.retro_pq_m,
#         args.retro_ivf_dim,
#         args.retro_nclusters,
#         args.retro_hnsw_m,
#         args.retro_pq_m,
#     )


def get_index_dir():
    """Create sub-directory for this index."""
    
    args = get_retro_args()

    # Directory path.
    index_dir_path = os.path.join(
        args.retro_workdir,
        "index",
        args.retro_index_ty,
        # get_index_str(),
        args.retro_index_str,
    )

    # Make directory.
    os.makedirs(index_dir_path, exist_ok = True)

    return index_dir_path


def get_training_data_dir():
    return os.path.join(get_index_dir(), "training_data_tmp")


def get_training_data_block_dir():
    return os.path.join(get_training_data_dir(), "blocks")


def get_training_data_block_paths():
    return sorted(glob.glob(get_training_data_block_dir() + "/*.hdf5"))


# def get_training_data_merged_path():
#     return os.path.join(get_training_data_dir(), "merged.hdf5")


# def get_training_data_merged():
#     with h5py.File(get_training_data_merged_path(), "r") as f:
#         shape = f["data"].shape

#         # # # >>> **debug**
#         # # # np.random.default_rng().standard_normal(size = 1, dtype = "f4")
#         # # np.random.default_rng().random(size = 1, dtype = "f4")
#         # # return np.random.rand(*shape).astype("f4")
#         # print_rank_0("rando merged.")
#         # return np.random.rand(int(1e6), shape[1]).astype("f4")
#         return np.zeros(shape, dtype = "f4")
#         # # # <<<

#         data = np.empty(shape, dtype = "f4")
#         block_size = 10000000
#         # >>>
#         # pbar = tqdm(range(0, shape[0], block_size))
#         # pbar.set_description("loading merged training data")
#         # for start_idx in pbar:
#         #     end_idx = min(shape[0], start_idx + block_size)
#         #     data[start_idx:end_idx] = f["data"][start_idx:end_idx]
#         # +++
#         for start_idx in range(0, shape[0], block_size):
#             print_rank_0("loading merged block %d / %d." % (
#                 int(start_idx / block_size),
#                 int(np.ceil(shape[0] / block_size)),
#             ))
#             end_idx = min(shape[0], start_idx + block_size)
#             data[start_idx:end_idx] = f["data"][start_idx:end_idx]
#         # <<<

#         print_rank_0("finished loading merged data.")

#         return data
# def get_training_data_merged():
#     # pax(0, {"block_paths": get_training_data_block_paths()})
#     from tools.bert_embedding.utils import load_data
#     from tools.retro.utils import Timer
#     return load_data(get_training_data_block_paths(), Timer())
def get_training_data_merged():

    args = get_retro_args()

    # Setup.
    block_paths = get_training_data_block_paths()
    ds_infos = get_indexed_dataset_infos()
    n_chunks_sampled = sum(d["n_chunks_sampled"] for d in ds_infos)

    # Initialize merged data.
    data = np.empty((n_chunks_sampled, args.retro_nfeats), dtype = "f4")
    # data.fill(0) # ... allocates 1.2TB, for real

    # Load data blocks.
    start_idx = 0
    pbar = tqdm(block_paths)
    for path_index, path in enumerate(pbar):
        pbar.set_description("mem %.0f gb, %.1f%%" % (
            psutil.virtual_memory()[3] / 1024**3,
            psutil.virtual_memory()[2],
        ))
        # t = time.time()
        with h5py.File(path, "r") as f:
            n_current = len(f["data"])
            data[start_idx:(start_idx+n_current)] = f["data"]
            start_idx += n_current
        # t = time.time() - t
        # if path_index % 50 == 0:
        #     print("load train block %d / %d ... %s sec, mem %.0f gb [ %.1f ]." % (
        #         path_index,
        #         len(block_paths),
        #         t,
        #         psutil.virtual_memory()[3] / 1024**3,
        #         psutil.virtual_memory()[2],
        #     ))

    # Verify.
    assert start_idx == n_chunks_sampled

    # pax(0, {
    #     # "block_paths" : block_paths,
    #     "ds_infos" : ds_infos,
    #     "n_chunks_sampled" : n_chunks_sampled,
    #     "data" : "%s / %s / %s" % (data.shape, data.dtype, str(data)),
    # })

    return data


def remove_training_data():
    if torch.distributed.get_rank() != 0:
        return
    raise Exception("ready to delete?")
    shutil.rmtree(get_training_data_dir())
