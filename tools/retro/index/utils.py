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


def get_index_dir():
    """Create sub-directory for this index."""
    
    args = get_retro_args()

    # Directory path.
    index_dir_path = os.path.join(
        args.retro_workdir,
        "index",
        args.retro_index_ty,
        args.retro_index_str,
    )

    # Make directory.
    os.makedirs(index_dir_path, exist_ok = True)

    return index_dir_path


def num_samples_to_block_ranges(num_samples):
    '''Split a range (length num_samples) into sequence of block ranges
    of size block_size.'''
    args = get_retro_args()
    block_size = args.retro_block_size
    start_idxs = list(range(0, num_samples, block_size))
    end_idxs = [min(num_samples, s + block_size) for s in start_idxs]
    ranges = list(zip(start_idxs, end_idxs))
    return ranges


def get_training_data_dir():
    return os.path.join(get_index_dir(), "training_data_tmp")


def get_training_data_paths():
    return sorted(glob.glob(get_training_data_dir() + "/*.hdf5"))


def get_training_data_merged():
    # >>>
    raise Exception("merge to np.memmap.")
    # <<<
    '''Merge embeddings into single dataset.'''

    args = get_retro_args()

    # Setup.
    block_paths = get_training_data_paths()
    ds_infos = get_indexed_dataset_infos()
    n_chunks_sampled = sum(d["n_chunks_sampled"] for d in ds_infos)

    # Initialize merged data.
    print("allocate training data array.")
    t = time.time()
    data = np.empty((n_chunks_sampled, args.retro_nfeats), dtype = "f4")
    data.fill(0) # ... allocates 1.2TB for real; *essential* for performance
    print("  time : %.3f sec." % (time.time() - t))

    # Load data blocks.
    print("load training data blocks.")
    start_idx = 0
    pbar = tqdm(block_paths)
    for path_index, path in enumerate(pbar):
        pbar.set_description("mem %.0f gb, %.1f%%" % (
            psutil.virtual_memory()[3] / 1024**3,
            psutil.virtual_memory()[2],
        ))
        with h5py.File(path, "r") as f:
            n_current = len(f["data"])
            data[start_idx:(start_idx+n_current)] = f["data"]
            start_idx += n_current

    # Verify.
    assert start_idx == n_chunks_sampled

    return data


def remove_training_data():
    '''Delete embeddings that were used for training.'''
    if torch.distributed.get_rank() != 0:
        return
    raise Exception("ready to delete?")
    shutil.rmtree(get_training_data_dir())
