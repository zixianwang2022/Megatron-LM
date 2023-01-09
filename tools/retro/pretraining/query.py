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
import faiss
import h5py
import numpy as np
import os
import time
import torch
from tqdm import tqdm

from megatron import get_retro_args, mpu, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.bert_embedding.utils import get_missing_blocks_by_rank
from tools.retro.db.utils import get_merged_train_dataset \
    as get_db_merged_train_dataset
from tools.retro.index.factory import IndexFactory
from tools.retro.index.utils import get_index_dir, num_samples_to_block_ranges
from tools.retro.utils import GPTToTextDataset

from .chunk_dataset import get_chunk_dataset_map

# >>>
from lutil import pax
# <<<


def get_index(chunk_db_dataset, ondisk = False):
    '''Read index from disk.'''

    args = get_retro_args()

    # Chunk db block ranges.
    n_db_chunks = len(chunk_db_dataset)
    dataset_block_ranges = num_samples_to_block_ranges(n_db_chunks)

    # Load index.
    index_wrapper = IndexFactory.get_index(args.retro_index_ty)
    index_dir = get_index_dir()
    added_index_path = index_wrapper.get_added_index_path()
    # >>>
    # pax(0, {"added_index_path": added_index_path})
    # <<<
    if ondisk:
        index = faiss.read_index(added_index_path, faiss.IO_FLAG_MMAP)
    else:
        index = faiss.read_index(added_index_path)

    # Search parameters.
    faiss.ParameterSpace().set_index_parameter(index, "efSearch",
                                               args.retro_ef_search)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe",
                                               args.retro_nprobe)

    return index


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def get_banned_chunk_map(chunk_db):
#     '''Get mapping of {(dataset_id,doc_id):[chunk_ids]}.'''

#     # Map docs to chunks.
#     print_rank_0("build doc-chunk-id-map.")
#     banned_chunk_map = defaultdict(set)
#     for chunk_id, (dataset_id, doc_id) in enumerate(tqdm(
#             chunk_db[:, :2],
#             "map banned chunks",
#             total = chunk_db.shape[0],
#     )):
#         banned_chunk_map[(dataset_id.item(), doc_id.item())].add(chunk_id)

#     return banned_chunk_map
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# def get_banned_chunk_map(chunk_db):
#     '''Get mapping of {(dataset_id,doc_id):[chunk_ids]}.'''

#     # print("copy chunk db.")
#     # # n = 1000000
#     # # n = 10000000
#     # n = 100000000
#     # t = time.time()
#     # np.copy(chunk_db[:n])
#     # print("end copy ... n %d, %.3f sec." % (n, time.time() - t))
#     # torch.distributed.barrier()
#     # exit()

#     n_chunks = chunk_db.shape[0]
#     block_size = 100000000

#     # Map docs to chunks.
#     print_rank_0("build doc-chunk-id-map.")
#     banned_chunk_map = defaultdict(set)
#     for chunk_start_id in range(0, n_chunks, block_size):

#         chunk_end_id = min(n_chunks, chunk_start_id + block_size)
#         sub_chunk_db = np.copy(chunk_db[chunk_start_id:chunk_end_id, :2])

#         # if chunk_start_id > 0:
#         #     pax({
#         #         "sub_chunk_db" : sub_chunk_db,
#         #         "n_chunks" : n_chunks,
#         #         "block_size" : block_size,
#         #         "chunk_start_id" : chunk_start_id,
#         #         "chunk_end_id" : chunk_end_id,
#         #     })

#         # continue

#         for rel_chunk_id, (dataset_id, doc_id) in enumerate(tqdm(
#                 sub_chunk_db,
#                 "map banned chunks, %d / %d" % (
#                     chunk_start_id // block_size,
#                     n_chunks // block_size,
#                 ),
#                 total = chunk_end_id - chunk_start_id,
#         )):
#             chunk_id = chunk_start_id + rel_chunk_id
#             banned_chunk_map[(dataset_id.item(), doc_id.item())].add(chunk_id)

#     pax(0, {"banned_chunk_map": banned_chunk_map})

#     return banned_chunk_map
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from concurrent.futures import (
    ProcessPoolExecutor,
    # ThreadPoolExecutor,
    as_completed,
)
import json
import zlib


# def get_partial_banned_chunk_map(chunk_db, chunk_id_range):

#     # sub_chunk_db = chunk_db[chunk_id_range[0]:chunk_id_range[1], :2]
#     sub_chunk_db = np.copy(chunk_db[chunk_id_range[0]:chunk_id_range[1], :2])

#     # Map docs to chunks.
#     banned_chunk_map = defaultdict(set)
#     for rel_chunk_id, (dataset_id, doc_id) in enumerate(tqdm(
#             sub_chunk_db,
#             "map banned chunks",
#             total = chunk_id_range[1] - chunk_id_range[0],
#     )):
#         chunk_id = chunk_id_range[0] + rel_chunk_id
#         banned_chunk_map[(dataset_id.item(), doc_id.item())].add(chunk_id)

#     return banned_chunk_map
# def get_partial_banned_chunk_map(thread_id, start_chunk_id, sub_chunk_db):

#     # Map docs to chunks.
#     banned_chunk_map = defaultdict(set)
#     for rel_chunk_id, (dataset_id, doc_id) in enumerate(tqdm(
#             sub_chunk_db,
#             "map banned docs, thread %d" % thread_id,
#             total = sub_chunk_db.shape[0],
#             # disable = thread_id % 8 == 0,
#     )):
#         chunk_id = start_chunk_id + rel_chunk_id
#         banned_chunk_map[(dataset_id.item(), doc_id.item())].add(chunk_id)

#     return banned_chunk_map
def get_partial_banned_chunk_map(proc_id,
                                 n_chunks,
                                 start_chunk_id,
                                 end_chunk_id,
                                 db_path):

    # >>>
    n_digits = int(np.ceil(np.log(n_chunks) / np.log(10)) + 1)
    output_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/corpus/banned_tmp/%s-%s.json" % (str(start_chunk_id).zfill(n_digits), str(end_chunk_id).zfill(n_digits))

    if os.path.exists(output_path):
        return proc_id, output_path
    # <<<

    # Chunk subset.
    with h5py.File(db_path) as f:
        # >>>
        sub_chunk_db = np.copy(f["chunks"][start_chunk_id:end_chunk_id, :2])
        # sub_chunk_db = np.copy( # .......... debuggggggging .....
        #     f["chunks"][start_chunk_id:start_chunk_id+1000000, :2])
        # <<<

    # Map docs to chunks.
    banned_chunk_map = defaultdict(list)
    for rel_chunk_id, (dataset_id, doc_id) in enumerate(tqdm(
            sub_chunk_db,
            "map banned docs, proc %d" % proc_id,
            total = sub_chunk_db.shape[0],
            # disable = proc_id % 8 == 0,
    )):
        chunk_id = start_chunk_id + rel_chunk_id
        banned_chunk_map["%d,%d" % (dataset_id.item(), doc_id.item())] \
            .append(chunk_id)

    # >>>
    # # Compress map.
    # compressed_map = zlib.compress(json.dumps(banned_chunk_map).encode())

    # return proc_id, compressed_map
    # +++
    with open(output_path, "w") as f:
        json.dump(banned_chunk_map, f)

    return proc_id, output_path
    # <<<


def get_banned_chunk_map(db_dataset):
    '''Get mapping of {(dataset_id,doc_id):[chunk_ids]}.'''

    # >>>
    # n_procs = 128
    n_procs = 32 # *
    # n_procs = 1
    # <<<

    n_chunks = db_dataset.chunks.shape[0]
    n_chunks_per_proc = max(1, int(np.ceil(n_chunks / n_procs)))
    chunk_id_starts = list(range(0, n_chunks, n_chunks_per_proc))
    chunk_id_ranges = [(s, min(n_chunks, s + n_chunks_per_proc))
                       for s in chunk_id_starts]
    
    print_rank_0("build doc-chunk-id-map.")
    with ProcessPoolExecutor(max_workers = n_procs) as executor:

        # Build partial chunk maps.
        futures = []
        for proc_id, (start_chunk_id, end_chunk_id) \
            in enumerate(chunk_id_ranges):
            futures.append(executor.submit(
                get_partial_banned_chunk_map,
                proc_id,
                n_chunks,
                start_chunk_id,
                end_chunk_id,
                db_dataset.db_path,
            ))

        # >>>
        # # Wait for processes to finish.
        # compressed_banned_chunk_map = {}
        # for finished_idx, future in enumerate(as_completed(futures)):
        #     print("finished %d / %d." % (finished_idx, n_procs))
        #     proc_id, proc_banned_chunk_map = future.result()
        #     compressed_banned_chunk_map[proc_id] = proc_banned_chunk_map
        # +++
        # Wait for processes to finish.
        banned_chunk_paths = []
        for finished_idx, future in enumerate(as_completed(futures)):
            print("finished %d / %d." % (finished_idx, n_procs))
            _, banned_chunk_path = future.result()
            banned_chunk_paths.append(banned_chunk_path)

        banned_chunk_paths.sort() # non-essential
        # <<<

        # >>>
        banned_chunk_map = {}
        for banned_chunk_path in tqdm(banned_chunk_paths,
                                      "load banned chunk paths"):
            with open(banned_chunk_path) as f:
                crnt_banned_chunk_map = json.load(f)
                print("... do something. ...")
        # <<<

        pax(0, {"banned_chunk_paths": banned_chunk_paths})

        # Merge partial maps into one.
        banned_chunk_map = defaultdict(set)
        for join_idx, (proc_id, proc_banned_chunk_map) \
            in enumerate(compressed_banned_chunk_map.items()):

            # De-compress.
            proc_banned_chunk_map = \
                json.loads(zlib.decompress(proc_banned_chunk_map).decode())

            # Merge.
            for key, chunk_ids in tqdm(
                    proc_banned_chunk_map.items(),
                    "join map %d / %d" % (
                        join_idx,
                        len(compressed_banned_chunk_map),
                    ),
            ):
                key = tuple(int(i) for i in key.split(","))
                banned_chunk_map[key].update(chunk_ids)

        pax({
            "num keys" : len(banned_chunk_map),
            "num chunks" : sum(len(v) for v in banned_chunk_map.values()),
        })

        return banned_chunk_map

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def embed_block(gpt_dataset, block, embedder):
    '''Embed block of chunks.'''
    text_block_dataset = torch.utils.data.Subset(
        GPTToTextDataset(gpt_dataset),
        range(*block["range"]),
    )
    return embedder.embed_text_dataset(text_block_dataset)


def query_embeddings(index, banned_chunk_map, chunk_id_range,
                     embeddings, sample_map, n_chunks_per_sample):
    '''Query neighbors of a block of embeddings.'''

    args = get_retro_args()

    # Query neighbor ids.
    print_rank_0("search.")
    t = time.time()
    assert index.ntotal > 0, "check we don't accidentally have an empty index."
    _, query_nbr_ids = index.search(embeddings, args.retro_nnbrs_query)
    print_rank_0("  time : %.3f sec." % (time.time() - t))

    # Banned neighbor ids.
    print_rank_0("get banned neighbor ids.")
    sample_banned_chunk_id_map = {}
    for sample_id, sample in sample_map.items():
        dataset_idx = sample["dataset_idx"].item()
        doc_ids = sample["doc_ids"].tolist()
        banned_chunk_ids = set()
        for doc_id in doc_ids:
            banned_chunk_ids.update(banned_chunk_map[(dataset_idx, doc_id)])
        sample_banned_chunk_id_map[sample_id] = banned_chunk_ids

    # Filter banned neighbor ids.
    print_rank_0("filter banned neighbor ids.")
    filtered_nbr_ids = np.full(
        shape = (len(query_nbr_ids), args.retro_nnbrs_target),
        fill_value = -1,
        dtype = "int64",
    )
    min_chunk_id, max_chunk_id = chunk_id_range
    for chunk_id in range(min_chunk_id, max_chunk_id):

        sample_id = chunk_id // n_chunks_per_sample

        # Get valid neighbors (!= -1).
        query_row = [ i for i in query_nbr_ids[chunk_id-min_chunk_id] if i >= 0 ]

        # Filter row.
        filtered_row = [i for i in query_row
                        if i not in sample_banned_chunk_id_map[sample_id]]
        filtered_row = filtered_row[:args.retro_nnbrs_target]
        filtered_row += \
            [-1] * (args.retro_nnbrs_target - len(filtered_row))
        filtered_nbr_ids[chunk_id-min_chunk_id] = filtered_row

    return query_nbr_ids, filtered_nbr_ids


def query_embedding_block(index, banned_chunk_map, chunk_id_range,
                          embeddings, sample_map, n_chunks_per_sample):

    query_nbr_ids = []
    filtered_nbr_ids = []

    partial_block_size = 1000
    for partial_start_idx in tqdm(
            range(0, len(embeddings), partial_block_size),
            "search",
    ):
        partial_end_idx = min(len(embeddings),
                              partial_start_idx + partial_block_size)
        partial_embeddings = embeddings[partial_start_idx:partial_end_idx]
        partial_chunk_id_range = (
            chunk_id_range[0] + partial_start_idx,
            chunk_id_range[0] + partial_end_idx,
        )
        partial_query_nbr_ids, partial_filtered_nbr_ids = \
            query_embeddings(index, banned_chunk_map, partial_chunk_id_range,
                             partial_embeddings, sample_map, n_chunks_per_sample)
        query_nbr_ids.append(partial_query_nbr_ids)
        filtered_nbr_ids.append(partial_filtered_nbr_ids)

    pax(0, {
        "query_nbr_ids" : query_nbr_ids,
        "filtered_nbr_ids" : filtered_nbr_ids,
    })

    return query_nbr_ids, filtered_nbr_ids


def query_block_neighbors(index, banned_chunk_map, chunk_dataset,
                          block, embedder):
    '''Query neighbors of a dataset block (i.e., range).'''

    args = get_retro_args()
    n_chunks_per_sample = chunk_dataset.n_chunks_per_sample

    # Sample map.
    sample_ids = sorted(list(set(chunk_id // n_chunks_per_sample
                                 for chunk_id in range(*block["range"]))))
    sample_map = {i:chunk_dataset.sample_dataset[i] for i in sample_ids}

    # Embed block.
    embeddings = embed_block(chunk_dataset, block, embedder)

    # Query embeddings.
    # >>>
    # _, filtered_nbr_ids = query_embeddings(
    _, filtered_nbr_ids = query_embedding_block(
        index, banned_chunk_map, block["range"],
        embeddings, sample_map,
        n_chunks_per_sample)
    # <<<

    # Save neighbors.
    print_rank_0("save neighbors.")
    os.makedirs(os.path.dirname(block["path"]), exist_ok = True)
    f = h5py.File(block["path"], "w")
    f.create_dataset("neighbors", data = filtered_nbr_ids)
    f.close()


def query_dataset_neighbors(index, banned_chunk_map,
                            prefix, chunk_dataset, nbr_dir,
                            embedder):
    '''Query neighbors of each chunk within a dataset.'''

    args = get_retro_args()

    def validate(f):
        assert f["neighbors"].shape[1] == args.retro_nnbrs_target
    n_missing_blocks, missing_nbr_blocks = get_missing_blocks_by_rank(
        nbr_dir,
        len(chunk_dataset),
        args.retro_block_size,
        validate = validate,
    )

    # Query each block.
    for block_index, block in enumerate(missing_nbr_blocks):

        if block is not None:

            # Progress.
            print_rank_0("query '%s' block %d / %d ... %s." % (
                prefix,
                block_index,
                len(missing_nbr_blocks),
                block["path"],
            ))

            # Query block neighbors.
            query_block_neighbors(index, banned_chunk_map,
                                  chunk_dataset, block, embedder)

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()


def query_pretraining_neighbors():
    '''Query pretraining datasets (train & valid).'''

    args = get_retro_args()

    # Num threads.
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        n_threads = 128
    elif world_size <= 2:
        n_threads = 64
    elif world_size <= 4:
        n_threads = 32
    elif world_size <= 8:
        n_threads = 16
    else:
        n_threads = 8

    # faiss.omp_set_num_threads(n_threads)
    faiss.omp_set_num_threads(16) # 32) # 4 procs/node
    # faiss.omp_set_num_threads(128) # 1 proc/node
    # print("n_threads = %d." % faiss.omp_get_max_threads())

    # Load chunk db dataset.
    print_rank_0("load chunk db dataset.")
    chunk_db_dataset = get_db_merged_train_dataset()

    # Load index, banned chunk ids, datasets.
    print_rank_0(" > get index.")
    # >>>
    # index = get_index(chunk_db_dataset)
    index = get_index(chunk_db_dataset, ondisk = True)
    # <<<

    print_rank_0(" > get banned doc-chunk id map.")
    # >>>
    # banned_chunk_map = get_banned_chunk_map(chunk_db_dataset.chunks)
    banned_chunk_map = get_banned_chunk_map(chunk_db_dataset)
    # <<<

    print_rank_0(" > get dataset map.")
    chunk_dataset_map = get_chunk_dataset_map()

    # Bert embedder.
    embedder = BertEmbedder(args.retro_bert_batch_size,
                            args.retro_bert_max_chunk_length,
                            args.bert_embedder_type)

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in chunk_dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        query_dataset_neighbors(index, banned_chunk_map,
                                prefix, info["data"], info["nbr_dir"],
                                embedder)
