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

# import argparse
import faiss
# from faiss import ParameterSpace
# import h5py
# import joblib
# import multiprocessing
# import numpy as np
# import os
# import psutil
# import time
import torch
# from tqdm import tqdm

from megatron import print_rank_0
# from tools.retrieval.utils import Timer
from tools.retrieval.index.factory import IndexFactory
from tools.retrieval.index.utils import get_index_workdir

from .dataset import get_dataset_map

# >>>
from lutil import pax
# <<<


def get_missing_neighbor_blocks(args, workdir, dataset_key):

    pax(0, {"workdir": workdir})

    embedding_dir = x

    n_chunks = len(dataset)

    # Block ranges.
    block_size = args.retro_block_size
    block_start_idxs = list(range(0, n_chunks, block_size))
    block_end_idxs = [ min(n_chunks, i + block_size) for i in block_start_idxs ]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # All block files (existing + missing).
    n_digits = int(np.ceil(np.log(n_chunks) / np.log(10)) + 1)
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
            pbar.set_description("verifying embedding block.")
            f = h5py.File(path, "r")
            try:
                assert f["data"].shape[1] == 1024
            except:
                raise Exception("delete block file.")
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
    print_seq("missing blocks [%d] : %s ... %s." % (
        len(rank_missing_block_items),
        str(rank_missing_block_items[0]["range"]),
        str(rank_missing_block_items[-1]["range"]) if rank_missing_block_items[-1] else str(rank_missing_block_items[-2]["range"]),
    ))
    # <<<

    return rank_missing_block_items


# def query_neighbors_single(args, workdir, dataset_key, dataset_path):
def query_dataset_neighbors(args, workdir, dataset_key, dataset_path):

    pax(0, {
        "workdir" : workdir,
        "dataset_key" : dataset_key,
        "dataset_path" : dataset_path,
    })

    missing_neighbor_blocks = get_missing_neighbor_blocks(args, workdir, ddd)


    # Load index.
    timer.push("load-index")
    index_wrapper = IndexFactory.get_index(args)
    index_dir = get_index_workdir(args)
    added_index_path = index_wrapper.get_added_index_path(None, index_dir)
    index = faiss.read_index(added_index_path)
    timer.pop()

    # pax(0, {
    #     "index_wrapper" : index_wrapper,
    #     "index" : index,
    #     "index_dir" : index_dir,
    #     "added_index_path" : added_index_path,
    # })

    if 0:
        ParameterSpace().set_index_parameter(index, "efSearch", args.efsearch)
        ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)


    for data_start in range(args.start, args.split):

        # print("Loading features...")
        timer.push("load-feats")
        # features to be queried
        f = h5py.File(args.feature_path, "r")
        features = f['feat']
        length = len(features)
        split_size = int(np.ceil(length / args.split))
        # >>>
        # start = split_size * args.start
        # end = min(split_size * (args.start + 1), length)
        # +++
        start = split_size * data_start
        end = min(split_size * (data_start + 1), length)
        # <<<
        print(f"Start Index: {start}, End Index: {end}, Split: {args.split}, Split size: {split_size}")
        data = np.copy(features[start:end])
        f.close()
        timer.pop()

        print("features dim:", features.shape)

        ## query
        # print("Searching kNN...")
        timer.push("search")
        # start = time.time()
        # k = args.k                             # we want to see 2000 nearest neighbors
        # D, I = index.search(data, k)         # sanity check
        D, I = index.search(data, args.k)         # sanity check
        # end = time.time()
        # print("time cost:", end - start)
        timer.pop()

        neighbors = np.zeros((len(data), args.target_k), 'uint64')

        def filter_neighbors(i):
            chunk_id = i + data_start * split_size
            sample_id = chunk_id // 32 + args.offset
            neighbors = I[i]
            banned_docs = doc_ids[sample_id]

            filtered_neighbors = []

            for neighbor in neighbors:
                if document_ids[neighbor] not in banned_docs:
                    filtered_neighbors.append(neighbor)
                if len(filtered_neighbors) == args.target_k:
                    return filtered_neighbors
            return filtered_neighbors

        pool = multiprocessing.Pool(args.workers)

        neighbor_ind = np.arange(len(data))
        delayed_neighbors = pool.imap(filter_neighbors, neighbor_ind, 25)

        timer.push("filter-nbrs")
        tot = 0
        # start = time.time()
        for i, filtered_neighbors in enumerate(tqdm(delayed_neighbors, total=len(neighbor_ind))):
            if len(filtered_neighbors) < args.target_k:
                filtered_neighbors += [-1] * (args.target_k - len(filtered_neighbors))
                tot += 1
            assert len(filtered_neighbors) == args.target_k
            neighbors[i] = filtered_neighbors
        # end = time.time()
        print("Number of neighbors < target k:", tot)
        # print("time cost:", end - start)
        timer.pop()

        # for i in tqdm(range(len(data))):
        #     filtered_neighbors = filter_neighbors(i)
        #     if len(filtered_neighbors) < args.target_k:
        #         filtered_neighbors += [-1] * (args.target_k - len(filtered_neighbors))
        #         tot += 1
        #     assert len(filtered_neighbors) == args.target_k
        #     neighbors[i] = filtered_neighbors
        #
        # print("Number of neighbors < target k:", tot)

        # if not args.out_path:
        #     out_path = f"{args.feature_path}_neighbors_start_{args.start}_split_{args.split}.hdf5"
        # else:
        #     out_path = args.out_path
        # out_path = f"{args.out_dir}/{args.out_prefix}__neighbors_start_{args.start}_split_{args.split}.hdf5"
        out_path = f"{args.out_dir}/{args.out_prefix}__neighbors_start_{data_start}_split_{args.split}.hdf5"
        print(f"Dumping to {out_path}")

        fout = h5py.File(out_path, "w")
        fout.create_dataset("neighbors", data=neighbors)
        fout.close()

        # >>>
        print("early exit.")
        break
        # <<<

    timer.print()


def query_neighbors(args, workdir, timer):

    # >>>
    # # Set num threads (torch.distributed reset it to 1).
    # # assert torch.distributed.get_rank() == 0
    # if torch.distributed.get_rank() != 0:
    #     return
    # faiss.omp_set_num_threads(64)
    # <<<

    # Query each (i.e., train, valid, test) dataset.
    dataset_map = get_dataset_map(args, workdir)
    for key, info in dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (key, len(info["data"])))
        query_dataset_neighbors(args, workdir, key, info["dir"]) # , info["data"])

