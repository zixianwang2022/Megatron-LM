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


def get_banned_chunk_map(chunk_db):
    '''Get mapping of {(dataset_id,doc_id):[chunk_ids]}.'''

    # Map docs to chunks.
    print_rank_0("build doc-chunk-id-map.")
    banned_chunk_map = defaultdict(set)
    for chunk_id, (dataset_id, doc_id) in \
        enumerate(tqdm(chunk_db[:, :2], "map banned chunks")):
        banned_chunk_map[(dataset_id.item(), doc_id.item())].add(chunk_id)

    return banned_chunk_map


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
    index = get_index(chunk_db_dataset)

    print_rank_0(" > get banned doc-chunk id map.")
    banned_chunk_map = get_banned_chunk_map(chunk_db_dataset.chunks)

    print_rank_0(" > get dataset map.")
    chunk_dataset_map = get_chunk_dataset_map()

    # Bert embedder.
    embedder = BertEmbedder(args.retro_bert_batch_size,
                            args.retro_bert_max_chunk_length)

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in chunk_dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        query_dataset_neighbors(index, banned_chunk_map,
                                prefix, info["data"], info["nbr_dir"],
                                embedder)
