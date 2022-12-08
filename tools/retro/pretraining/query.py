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
# from faiss import ParameterSpace
import h5py
import numpy as np
import os
import torch
from tqdm import tqdm

from megatron import get_retro_args, mpu, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.bert_embedding.utils import get_missing_blocks_by_rank
from tools.retro.db.utils import get_merged_train_dataset \
    as get_db_merged_train_dataset
from tools.retro.index.factory import IndexFactory
from tools.retro.index.indexes.faiss_par_add import get_dataset_block_ranges
from tools.retro.index.utils import get_index_dir
from tools.retro.utils import GPTToTextDataset

from .chunk_dataset import get_gpt_chunk_dataset_map

# >>>
from lutil import pax, print_seq, shorten as shorten_str
# <<<


def get_index(chunk_db_dataset, ondisk = False):

    args = get_retro_args()

    # Chunk db block ranges.
    n_db_chunks = len(chunk_db_dataset)
    dataset_block_ranges = get_dataset_block_ranges(n_db_chunks)

    # Load index.
    index_wrapper = IndexFactory.get_index(args.retro_index_ty)
    index_dir = get_index_dir()
    added_index_path = index_wrapper.get_added_index_path(dataset_block_ranges,
                                                          index_dir)
    # pax(0, {"added_index_path": added_index_path})
    if ondisk:
        index = faiss.read_index(added_index_path, faiss.IO_FLAG_MMAP)
    else:
        index = faiss.read_index(added_index_path)

    # Search parameters.
    # if 1:
    faiss.ParameterSpace().set_index_parameter(index, "efSearch",
                                               args.retro_ef_search)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe",
                                               args.retro_nprobe)

    # pax(0, {
    #     "index_wrapper" : index_wrapper,
    #     "index" : index,
    #     "index_dir" : index_dir,
    #     "added_index_path" : added_index_path,
    #     "ef search" : args.retro_ef_search,
    #     "nprobe" : args.retro_nprobe,
    # })

    return index


def get_banned_chunk_map(chunk_db):

    # Map docs to chunks.
    print_rank_0("build doc-chunk-id-map.")
    banned_chunk_map = defaultdict(set)
    for chunk_id, (dataset_id, doc_id) in \
        enumerate(tqdm(chunk_db[:, :2], "map banned chunks")):
        banned_chunk_map[(dataset_id.item(), doc_id.item())].add(chunk_id)

    # >>>
    # n_empties = len([v for v in banned_chunk_map.values() if not v])
    # pax(0, {
    #     # "chunk_db_path" : chunk_db_path,
    #     "banned_chunk_map" : {
    #         d : "%d / %s" % (len(cs), shorten_str(str(cs), 50))
    #         for i, (d, cs) in enumerate(banned_chunk_map.items())
    #         if i < 10 or i >= len(banned_chunk_map) - 10
    #     },
    #     "banned_chunk_map / len" : len(banned_chunk_map),
    #     "banned_chunk_map / (0, 836728)" : banned_chunk_map[(0, 836728)],
    #     "banned_chunk_map / (0, 836728) / bool" : bool(banned_chunk_map[(0, 836728)]),
    #     "n_empties" : n_empties,
    # })
    # <<<

    return banned_chunk_map


def embed_block(gpt_dataset, block, embedder):
    text_block_dataset = torch.utils.data.Subset(
        GPTToTextDataset(gpt_dataset),
        range(*block["range"]),
    )
    return embedder.embed_text_dataset(text_block_dataset, len(gpt_dataset))


# def query_block_neighbors(index, banned_chunk_map, dataset, block, embedder):

#     # raise Exception("loading 'added' index?")

#     args = get_retro_args()

#     # Embed block.
#     embeddings = embed_block(dataset, block, embedder)

#     # Query neighbor ids.
#     # >>>
#     from tools.retro.utils import Timer
#     timer = Timer()
#     timer.push("search")
#     # <<<
#     print_rank_0("search.")
#     assert index.ntotal > 0, "check we don't accidentally have an empty index."
#     _, query_nbr_ids = index.search(embeddings, args.retro_nnbrs_query)
#     # >>>
#     timer.pop()
#     # <<<

#     # Banned neighbor ids.
#     print_rank_0("get banned neighbor ids.")
#     sample_ids = sorted(list(set(chunk_id // dataset.n_chunks_per_seq
#                                  for chunk_id in range(*block["range"]))))
#     sample_banned_chunk_id_map = {}
#     for sample_id in sample_ids:
#         sample = dataset.seq_dataset[sample_id]
#         dataset_idx = sample["dataset_idx"].item()
#         doc_ids = sample["doc_ids"].tolist()
#         banned_chunk_ids = set()
#         for doc_id in doc_ids:
#             banned_chunk_ids.update(banned_chunk_map[(dataset_idx, doc_id)])
#         sample_banned_chunk_id_map[sample_id] = banned_chunk_ids

#     # pax(0, {"sample_banned_chunk_id_map": {
#     #     k : "%d / %s" % (len(v), shorten_str(str(v), 50))
#     #     for k, v in sample_banned_chunk_id_map.items()
#     # }})

#     # Filter banned neighbor ids.
#     print_rank_0("filter banned neighbor ids.")
#     filtered_nbr_ids = np.full(
#         shape = (len(query_nbr_ids), args.retro_nnbrs_target),
#         fill_value = -1,
#         dtype = "int64",
#     )
#     min_chunk_id, max_chunk_id = block["range"]
#     for chunk_id in range(min_chunk_id, max_chunk_id):

#         sample_id = chunk_id // dataset.n_chunks_per_seq

#         # Get valid neighbors (!= -1).
#         query_row = [ i for i in query_nbr_ids[chunk_id-min_chunk_id] if i >= 0 ]

#         # Filter row.
#         filtered_row = [i for i in query_row
#                         if i not in sample_banned_chunk_id_map[sample_id]]
#         filtered_row = filtered_row[:args.retro_nnbrs_target]
#         filtered_row += \
#             [-1] * (args.retro_nnbrs_target - len(filtered_row))
#         filtered_nbr_ids[chunk_id-min_chunk_id] = filtered_row

#     # Save neighbors.
#     if block["path"]:
#         print_rank_0("save neighbors.")
#         os.makedirs(os.path.dirname(block["path"]), exist_ok = True)
#         f = h5py.File(block["path"], "w")
#         f.create_dataset("neighbors", data = filtered_nbr_ids)
#         f.close()

#     return filtered_nbr_ids
def query_embeddings(index, banned_chunk_map, chunk_id_range,
                     embeddings, sample_map, n_chunks_per_seq):

    args = get_retro_args()

    # pax({"sample_map": sample_map})

    # Query neighbor ids.
    # >>>
    from tools.retro.utils import Timer
    timer = Timer()
    timer.push("search")
    # <<<
    print_rank_0("search.")
    assert index.ntotal > 0, "check we don't accidentally have an empty index."
    _, query_nbr_ids = index.search(embeddings, args.retro_nnbrs_query)
    # >>>
    timer.pop()
    # <<<

    # pax({"query_nbr_ids": query_nbr_ids})

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

    # pax(0, {"sample_banned_chunk_id_map": {
    #     k : "%d / %s" % (len(v), shorten_str(str(v), 50))
    #     for k, v in sample_banned_chunk_id_map.items()
    # }})

    # Filter banned neighbor ids.
    print_rank_0("filter banned neighbor ids.")
    filtered_nbr_ids = np.full(
        shape = (len(query_nbr_ids), args.retro_nnbrs_target),
        fill_value = -1,
        dtype = "int64",
    )
    min_chunk_id, max_chunk_id = chunk_id_range
    for chunk_id in range(min_chunk_id, max_chunk_id):

        sample_id = chunk_id // n_chunks_per_seq

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


def query_block_neighbors(index, banned_chunk_map, dataset, block, embedder):

    args = get_retro_args()

    # Sample map.
    sample_ids = sorted(list(set(chunk_id // n_chunks_per_seq
                                 for chunk_id in range(*block["range"]))))
    sample_map = {i:dataset.seq_dataset[i] for i in sample_ids}
    pax(0, {"sample_map": sample_map})

    # Embed block.
    embeddings = embed_block(dataset, block, embedder)

    # Query embeddings.
    _, filtered_nbr_ids = query_embeddings(
        index, banned_chunk_map, block["range"],
        embeddings, sample_map,
        dataset.n_chunks_per_seq)

    # Save neighbors.
    print_rank_0("save neighbors.")
    os.makedirs(os.path.dirname(block["path"]), exist_ok = True)
    f = h5py.File(block["path"], "w")
    f.create_dataset("neighbors", data = filtered_nbr_ids)
    f.close()


def query_dataset_neighbors(index, banned_chunk_map,
                            # prefix, embed_dir, nbr_dir, dataset):
                            prefix, dataset, nbr_dir,
                            embedder):

    args = get_retro_args()

    # missing_nbr_blocks = get_missing_neighbor_blocks(embed_dir, nbr_dir)
    def validate(f):
        assert f["neighbors"].shape[1] == args.retro_nnbrs_target
    n_missing_blocks, missing_nbr_blocks = get_missing_blocks_by_rank(
        nbr_dir,
        len(dataset),
        args.retro_block_size,
        validate = validate,
    )

    # >>>
    # if True or prefix != "train":
    #     pax(0, {
    #         "prefix" : prefix,
    #         "dataset / len" : len(dataset),
    #         "nbr_dir" : nbr_dir,
    #         "n_missing_blocks" : n_missing_blocks,
    #         "missing_nbr_blocks / sample" : missing_nbr_blocks[:10],
    #     })
    # <<<

    for block_index, block in enumerate(missing_nbr_blocks):

        if block is not None:

            # Progress.
            print_rank_0("query '%s' block %d / %d ... %s." % (
                prefix,
                block_index,
                len(missing_nbr_blocks),
                # block["nbr_path"],
                block["path"],
            ))

            # Query block neighbors.
            query_block_neighbors(index, banned_chunk_map,
                                  dataset, block, embedder)

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()


def query_pretraining_neighbors(timer):

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

    # >>>
    # faiss.omp_set_num_threads(n_threads)
    faiss.omp_set_num_threads(16) # 32) # 4 procs/node
    # faiss.omp_set_num_threads(128) # 1 proc/node
    # print("n_threads = %d." % faiss.omp_get_max_threads())
    # <<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # print_rank_0(" > get dataset map.")
    # chunk_dataset_map = get_gpt_chunk_dataset_map()
    # pax(0, {
    #     k : {
    #         "nbr_dir" : d["nbr_dir"],
    #         "dataset" : d["data"],
    #         "chunk len" : len(d["data"]),
    #         "seq len" : len(d["data"].seq_dataset),
    #     }
    #     for k, d in chunk_dataset_map.items()
    # })
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Load chunk db dataset.
    print_rank_0("load chunk db dataset.")
    chunk_db_dataset = get_db_merged_train_dataset()

    # Load index, banned chunk ids, datasets.
    print_rank_0(" > get index.")
    # >>>
    index = get_index(chunk_db_dataset)
    # index = faiss.read_index("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/empty.faissindex")
    # <<<

    print_rank_0(" > get banned doc-chunk id map.")
    banned_chunk_map = get_banned_chunk_map(chunk_db_dataset.chunk_db)

    print_rank_0(" > get dataset map.")
    chunk_dataset_map = get_gpt_chunk_dataset_map()

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
