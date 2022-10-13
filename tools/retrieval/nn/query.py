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
# import faiss
# from faiss import ParameterSpace
# import h5py
# import joblib
# import multiprocessing
# import numpy as np
# import os
# import psutil
# import time
# from tqdm import tqdm

# from tools.retrieval.utils import Timer


# timer = Timer()

# timer.push("setup")
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--index-path', type=str, default='',
#                     help='path to load the faiss index')
# parser.add_argument('--feature-path', type=str, default='',
#                     help='path to load the chunk feature hdf5')
# parser.add_argument('--doc-path', type=str, default='',
#                     help='path to load the banned doc id pkl')
# parser.add_argument('--chunk-path', type=str, default='',
#                     help='path to load the chunk doc id pkl')
# # parser.add_argument('--out-path', type=str, default='',
# #                     help='data path to load the jsonl')
# parser.add_argument('--out-dir', type=str, required=True,
#                     help='output dir.')
# parser.add_argument('--out-prefix', type=str, required=True,
#                     help='output prefix.')
# parser.add_argument('--start', type=int, default=0,
#                    help='Number of worker processes to launch')
# parser.add_argument('--offset', type=int, default=0,
#                    help='Number of worker processes to launch')
# parser.add_argument('--split', type=int, default=2,
#                    help='Number of splits of input features')
# parser.add_argument('--target-k', type=int, default=200,
#                    help='Number of neighbors to dump')
# parser.add_argument('--k', type=int, default=2000,
#                    help='Number of neighbors to retrieve')
# parser.add_argument('--workers', type=int, default=20,
#                    help='Number of worker processes to launch')
# # faiss index
# parser.add_argument('--efsearch', type=int, default=256,
#                    help='Number of worker processes to launch')
# parser.add_argument('--nprobe', type=int, default=65536,
#                    help='Number of worker processes to launch')
# parser.add_argument("--zfs", action='store_true', default=False,
#                    help='Use zfs data.')

# args = parser.parse_args()

# process = psutil.Process(os.getpid())
# print(process.memory_info().rss / 1024 / 1024 / 1024)  # in bytes

# ngpus = faiss.get_num_gpus()
# print("number of GPUs:", ngpus)

# # root1 = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/"
# # root2 = "/home/dcg-adlr-boxinw-data/processed_data/chunks/"
# timer.pop()

# Load index.
timer.push("load-index")
index = faiss.read_index(args.index_path)
timer.pop()

## load banned document id list
# doc_ids = joblib.load('/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Wikipedia-shuf/Wikipedia_en_ftfy_id_shuf_text_document.doc_ids.pkl')
timer.push("load-banned-doc-ids")
doc_ids = joblib.load(args.doc_path)
timer.pop()

## load chunk id
timer.push("load-chunk-ids")
f = h5py.File(args.chunk_path, "r")
document_ids = np.copy(f['document_id'])
f.close()
timer.pop()

ParameterSpace().set_index_parameter(index, "efSearch", args.efsearch)
ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)

print("efSearch", args.efsearch)
print("nprobe", args.nprobe)

# for data_start in range(args.split):
#     args.start = data_start
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
