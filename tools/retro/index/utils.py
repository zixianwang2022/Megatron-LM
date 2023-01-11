# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import glob
import h5py
import numpy as np
import os
import psutil
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
    return os.path.join(get_index_dir(), "train_tmp")


def get_training_data_paths():
    return sorted(glob.glob(get_training_data_dir() + "/*.hdf5"))


def get_added_codes_dir():
    return os.path.join(get_index_dir(), "add_tmp")


def get_added_code_paths():
    return sorted(glob.glob(get_added_codes_dir() + "/*.hdf5"))


# >>>
# def get_training_data_merged():
#     # >>>
#     raise Exception("merge to np.memmap.")
#     # <<<
#     '''Merge embeddings into single dataset.'''

#     args = get_retro_args()

#     # Setup.
#     block_paths = get_training_data_paths()
#     ds_infos = get_indexed_dataset_infos()
#     n_chunks_sampled = sum(d["n_chunks_sampled"] for d in ds_infos)

#     # Initialize merged data.
#     print("allocate training data array.")
#     t = time.time()
#     data = np.empty((n_chunks_sampled, args.retro_nfeats), dtype = "f4")
#     data.fill(0) # ... allocates 1.2TB for real; *essential* for performance
#     print("  time : %.3f sec." % (time.time() - t))

#     # Load data blocks.
#     print("load training data blocks.")
#     start_idx = 0
#     pbar = tqdm(block_paths)
#     for path_index, path in enumerate(pbar):
#         pbar.set_description("mem %.0f gb, %.1f%%" % (
#             psutil.virtual_memory()[3] / 1024**3,
#             psutil.virtual_memory()[2],
#         ))
#         with h5py.File(path, "r") as f:
#             n_current = len(f["data"])
#             data[start_idx:(start_idx+n_current)] = f["data"]
#             start_idx += n_current

#     # Verify.
#     assert start_idx == n_chunks_sampled

#     return data
# +++
# from lutil import pax

# def get_training_data_merged():
#     '''Merge embeddings into single dataset.'''

#     args = get_retro_args()
#     # >>>
#     load_ratio = 1.
#     # load_ratio = 2.5 / 3
#     # load_ratio = 0.1 / 3
#     # <<<

#     # Compute num samples.
#     block_paths = get_training_data_paths()
#     n_chunks_sampled = 0
#     for path in tqdm(block_paths, "compute n_chunks_sampled"):
#         with h5py.File(path, "r") as f:
#             n_chunks_sampled += int(load_ratio * f["data"].shape[0])

#     # >>>
#     # pax(0, {"n_chunks_sampled": n_chunks_sampled})
#     # <<<

#     # Initialize merged data.
#     print("allocate training data array.")
#     t = time.time()
#     data = np.empty((n_chunks_sampled, args.retro_nfeats), dtype = "f4")
#     data.fill(0) # ... allocates 1.2TB for real; *essential* for performance
#     print("  time : %.3f sec. (n %d)" % (time.time() - t, n_chunks_sampled))

#     # Load data blocks.
#     print("load training data blocks.")
#     start_idx = 0
#     pbar = tqdm(block_paths)
#     for path_index, path in enumerate(pbar):
#         pbar.set_description("mem %.0f gb, %.1f%%" % (
#             psutil.virtual_memory()[3] / 1024**3,
#             psutil.virtual_memory()[2],
#         ))
#         with h5py.File(path, "r") as f:
#             # n_current = len(f["data"])
#             n_current = int(load_ratio * f["data"].shape[0])
#             data[start_idx:(start_idx+n_current)] = f["data"][:n_current]
#             start_idx += n_current

#     # Verify.
#     assert start_idx == n_chunks_sampled

#     return data
# +++
import concurrent
import gc

from lutil import pax

# def get_block_path_groups():
# def get_training_data_groups():
def get_training_data_group_infos():

    args = get_retro_args()

    block_paths = get_training_data_paths()
    max_group_size = args.retro_index_train_block_size

    groups = []
    group = []
    group_size = 0
    for block_path in block_paths:
        with h5py.File(block_path) as f:
            block_size = f["data"].shape[0]
        group.append(block_path)
        group_size += block_size

        if group_size >= max_group_size:
            groups.append({
                "paths" : group,
                "size" : group_size,
            })
            group = []
            group_size = 0
    if group:
        groups.append({
            "paths" : group,
            "size" : group_size,
        })

    # pax(0, {
    #     "groups" : groups,
    #     "groups / 0" : groups[0],
    #     "groups / 0 / block paths" : groups[0]["paths"],
    #     "total group size" : sum(g["size"] for g in groups),
    # })

    return groups
    

def load_training_block(path, load_ratio):
    with h5py.File(path) as f:
        n_load = int(load_ratio * f["data"].shape[0])
        return np.copy(f["data"][:n_load])


def load_training_group(executor, group_info, load_ratio):

    # Launch threads to load block data.
    futures = []
    for path in group_info["paths"]:
        futures.append(executor.submit(load_training_block, path, load_ratio))

    # Collect block data.
    block_datas = []
    for future in futures:
        block_datas.append(future.result())

    # Concatenate blocks.
    group_data = np.concatenate(block_datas, axis = 0)

    # Verify.
    # assert group_data.shape[0] == group_info["size"]

    # Garbage collect (likely useless).
    for d in block_datas:
        del d
    gc.collect()

    # pax(0, {
    #     "group_info" : group_info,
    #     "group_data / shape" : str(group_data.shape),
    #     "group_data" : str(group_data),
    # })

    return group_data


def get_training_data_merged():
    '''Merge embeddings into single dataset.'''

    args = get_retro_args()

    # Setup.
    # block_paths = get_training_data_paths()
    ds_infos = get_indexed_dataset_infos()
    n_chunks_sampled = sum(d["n_chunks_sampled"] for d in ds_infos)

    # >>>
    # load_ratio = 1. # [ bad ]
    # load_ratio = 2.8 / 3 # [ timeout ]
    # load_ratio = 2.5 / 3 # [ timeout ]
    load_ratio = 2.0 / 3 # [ success ]
    # load_ratio = 0.1 / 3
    # <<<

    # Initialize merged data.
    print("allocate training data array.")
    t = time.time()
    data = np.empty((n_chunks_sampled, args.retro_nfeats), dtype = "f4")
    # data.fill(0) # ... allocates 1.2TB for real; *essential* for performance
    print("  time : %.3f sec." % (time.time() - t))

    # Data groups (minimizing fragmentation).
    group_infos = get_training_data_group_infos()

    # # Load data blocks.
    # print("load training data blocks.")
    # start_idx = 0
    # pbar = tqdm(block_groups)
    # for block_group in pbar:

    #     pbar.set_description("mem %.0f gb, %.1f%%" % (
    #         psutil.virtual_memory()[3] / 1024**3,
    #         psutil.virtual_memory()[2],
    #     ))

    #     block_paths = block_group["paths"]
    #     group_size = block_group["size"]
    #     group_data = np.empty((group_size, args.retro_nfeats), dtype = "f4")
    #     # group_data.fill(0)
    #     group_start_idx = 0
    #     for block_path in block_paths:
    #         with h5py.File(block_path) as f:
    #             n_current = len(f["data"])
    #             group_data[group_start_idx:(group_start_idx+n_current)] = \
    #                 f["data"]
    #             group_start_idx += n_current
    #     assert group_start_idx == group_size
    #     data[start_idx:(start_idx+group_size)] = group_data
    #     start_idx += group_size
    #     # pax(0, {"group_data": group_data})
    n_threads = max(len(group["paths"]) for group in group_infos)
    # pax(0, {"n_threads": n_threads})
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:

        # Load data blocks.
        print("load training data blocks.")
        start_idx = 0
        pbar = tqdm(group_infos)
        for group_info in pbar:

            pbar.set_description("mem %.0f gb, %.1f%%" % (
                psutil.virtual_memory()[3] / 1024**3,
                psutil.virtual_memory()[2],
            ))

            group_data = load_training_group(executor, group_info, load_ratio)
            data[start_idx:(start_idx+len(group_data))] = group_data
            start_idx += len(group_data)

            # Garbage collect (likely useless).
            del group_data
            gc.collect()

        # >>>
        # Handle load ratio <1.
        data = data[:start_idx]
        print(">>>>>> data.shape = %s." % str(data.shape))
        # <<<

        # Verify.
        # assert start_idx == n_chunks_sampled

    # pax(0, {
    #     "data" : str(data),
    #     "data / shape" : str(data.shape),
    #     "data / dtype" : str(data.dtype),
    # })

    return data
# <<<
