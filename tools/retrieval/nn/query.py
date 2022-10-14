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

# from tools.retrieval.utils import Timer
from tools.retrieval.index.factory import IndexFactory
from tools.retrieval.index.utils import get_index_workdir

from .dataset import get_dataset_map

# >>>
from lutil import pax
# <<<


def query_neighbors(args, workdir, timer):

    # Set num threads (torch.distributed reset it to 1).
    # assert torch.distributed.get_rank() == 0
    if torch.distributed.get_rank() != 0:
        return
    faiss.omp_set_num_threads(64)

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


    dataset_map = get_dataset_map(args, workdir)

    pax(0, {"dataset_map": dataset_map})

    
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
