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
from faiss import ParameterSpace
import h5py
import numpy as np
import os
import torch

from megatron import get_args, mpu, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.bert_embedding.utils import get_missing_blocks_by_rank
from tools.retro.db.utils import get_full_merged_dataset \
    as get_db_full_merged_dataset
from tools.retro.index.factory import IndexFactory
from tools.retro.index.indexes.faiss_par_add import get_dataset_block_ranges
from tools.retro.index.utils import get_index_dir
from tools.retro.utils import GPTToTextDataset

from .chunk_dataset import get_gpt_chunk_dataset_map

# >>>
from lutil import pax, print_seq, shorten as shorten_str
# <<<


def get_index(chunk_db_dataset):

    args = get_args()

    # Chunk db block ranges.
    n_db_chunks = len(chunk_db_dataset)
    dataset_block_ranges = get_dataset_block_ranges(n_db_chunks)

    # Load index.
    index_wrapper = IndexFactory.get_index(args.retro_index_ty)
    index_dir = get_index_dir()
    added_index_path = index_wrapper.get_added_index_path(dataset_block_ranges,
                                                          index_dir)
    # pax(0, {"added_index_path": added_index_path})
    index = faiss.read_index(added_index_path)

    # Search parameters.
    if 0:
        ParameterSpace().set_index_parameter(index, "efSearch", args.retro_ef_search)
        ParameterSpace().set_index_parameter(index, "nprobe", args.retro_nprobe)

    # pax(0, {
    #     "index_wrapper" : index_wrapper,
    #     "index" : index,
    #     "index_dir" : index_dir,
    #     "added_index_path" : added_index_path,
    # })

    return index


# def get_banned_chunk_map(chunk_db):

#     # # Load chunk db.
#     # print_rank_0("load chunk db.")
#     # chunk_db_path = get_db_info_map()["full"]["db_path"]
#     # f = h5py.File(chunk_db_path, "r")
#     # doc_ids = np.copy(f["chunks_valid"][:, 0]).tolist()
#     # dataset_offsets = np.copy(f["dataset_offsets_valid"]).tolist()
#     # f.close()

#     # dataset_ids = dataset_offsets_to_ids(dataset_offsets)
#     # assert len(doc_ids) == len(dataset_ids)

#     # Get db dataset ids, doc ids.
#     dataset_ids = chunk_db[:, 0].tolist()
#     doc_ids = chunk_db[:, 1].tolist()
#     # pax(0, {"dataset_ids": dataset_ids, "doc_ids": doc_ids})

#     # >>>
#     # pax(0, {
#     #     "doc_ids" : "%d / %s" % (len(doc_ids), str(doc_ids)),
#     #     "dataset_offsets": "%d / %s"%(len(dataset_offsets), str(dataset_offsets)),
#     #     "dataset_ids" : "%d / %s" % (len(dataset_ids), str(dataset_ids)),
#     #     "doc_ids / 836728" : [
#     #         (dataset_ids[i], doc_id)
#     #         for i, doc_id in enumerate(doc_ids)
#     #         if doc_id == 836728
#     #     ],
#     # })
#     # <<<

#     # Map docs to chunks.
#     print_rank_0("build doc-chunk-id-map.")
#     banned_chunk_map = defaultdict(set)
#     # banned_chunk_map = {}
#     for chunk_id, doc_id in enumerate(doc_ids):
#         if chunk_id % 10000000 == 0:
#             print_rank_0("mapping banned chunks, %.0f%%." %
#                          (100 * chunk_id / len(doc_ids)))
#         dataset_id = dataset_ids[chunk_id]
#         banned_chunk_map[(dataset_id, doc_id)].add(chunk_id)

#     # >>>
#     # n_empties = len([v for v in banned_chunk_map.values() if not v])
#     n_empties = len([v for v in banned_chunk_map.values() if len(v) == 0])
#     pax(0, {
#         # "chunk_db_path" : chunk_db_path,
#         "banned_chunk_map" : {
#             d : "%d / %s" % (len(cs), shorten_str(str(cs), 50))
#             for i, (d, cs) in enumerate(banned_chunk_map.items())
#             if i < 10 or i >= len(banned_chunk_map) - 10
#         },
#         "banned_chunk_map / (0, 836728)" : banned_chunk_map[(0, 836728)],
#         "banned_chunk_map / (0, 836728) / bool" : bool(banned_chunk_map[(0, 836728)]),
#         "n_empties" : n_empties,
#     })
#     # <<<

#     return banned_chunk_map
def get_banned_chunk_map(chunk_db):

    # Map docs to chunks.
    print_rank_0("build doc-chunk-id-map.")
    banned_chunk_map = defaultdict(set)
    for chunk_id, (dataset_id, doc_id) in enumerate(chunk_db[:, :2]):
        if chunk_id % 10000000 == 0:
            print_rank_0("mapping banned chunks, %.0f%%." %
                         (100 * chunk_id / len(chunk_db)))
        if dataset_id == 0 and doc_id == 836728:
            raise Exception("hi.")
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


def query_block_neighbors(index, banned_chunk_map, dataset, block, embedder):

    args = get_args()

    # Embed block.
    embeddings = embed_block(dataset, block, embedder)

    # Query neighbor ids.
    print_rank_0("search.")
    _, query_nbr_ids = index.search(embeddings, args.retro_nnbrs_query)

    # Banned neighbor ids.
    print_rank_0("get banned neighbor ids.")
    sample_ids = sorted(list(set(chunk_id // dataset.n_chunks_per_seq
                                 for chunk_id in range(*block["range"]))))
    sample_banned_chunk_id_map = {}
    for sample_id in sample_ids:
        sample = dataset.seq_dataset[sample_id]
        dataset_idx = sample["dataset_idx"].item()
        doc_ids = sample["doc_ids"].tolist()
        banned_chunk_ids = set()
        for doc_id in doc_ids:
            # banned_chunk_ids.update(
            #     banned_chunk_map.get((dataset_idx, doc_id), set()))

            # >>>

            current_chunk_ids = banned_chunk_map[(dataset_idx, doc_id)]
            # >>>
            # if not current_chunk_ids:
            #     pax(0, {
            #         "sample_id" : sample_id,
            #         "sample" : sample,
            #         "dataset_idx" : dataset_idx,
            #         "doc_id" : doc_id,
            #         "current_chunk_ids" : current_chunk_ids,
            #         "doc_id matches" :
            #         [ k for k in banned_chunk_map if k[1] == doc_id ],
            #     })

            # if current_chunk_ids:
            #     pax({"current_chunk_ids": "%d / %s." % (
            #         len(current_chunk_ids),
            #         str(current_chunk_ids),
            #     )})

            # **note**: when debugging w/ sampled db, not full doc_id coverage.
            # assert current_chunk_ids, "should be >=1 chunk_id."
            # <<<
            banned_chunk_ids.update(current_chunk_ids)
            # <<<
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
    min_chunk_id, max_chunk_id = block["range"]
    for chunk_id in range(min_chunk_id, max_chunk_id):

        sample_id = chunk_id // dataset.n_chunks_per_seq

        # Get valid neighbors (!= -1).
        query_row = [i for i in query_nbr_ids[chunk_id-min_chunk_id]
                     if i>=0]

        # Filter row.
        filtered_row = [i for i in query_row
                        if i not in sample_banned_chunk_id_map[sample_id]]
        filtered_row = filtered_row[:200]
        filtered_row += \
            [-1] * (args.retro_nnbrs_target - len(filtered_row))
        filtered_nbr_ids[chunk_id-min_chunk_id] = filtered_row

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

    args = get_args()

    # missing_nbr_blocks = get_missing_neighbor_blocks(embed_dir, nbr_dir)
    def validate(f):
        assert f["neighbors"].shape[1] == args.retro_nnbrs_query
    n_missing_blocks, missing_nbr_blocks = get_missing_blocks_by_rank(
        nbr_dir,
        len(dataset),
        args.retro_block_size,
        validate = validate,
    )
    # pax(0, {
    #     "nbr_dir" : nbr_dir,
    #     "n_missing_blocks" : n_missing_blocks,
    #     "missing_nbr_blocks / sample" : missing_nbr_blocks[:10],
    # })

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

    args = get_args()

    # >>>
    # Set num threads (torch.distributed reset it to 1).
    # assert torch.distributed.get_rank() == 0
    # if torch.distributed.get_rank() != 0:
    #     return
    # faiss.omp_set_num_threads(64)
    faiss.omp_set_num_threads(8)
    # <<<

    # Load chunk db dataset.
    print_rank_0("load chunk db dataset.")
    chunk_db_dataset = get_db_full_merged_dataset()

    # Load index, banned chunk ids, datasets.
    print_rank_0(" > get index.")
    index = get_index(chunk_db_dataset)
    # pax(0, {"index": index})

    print_rank_0(" > get banned doc-chunk id map.")
    banned_chunk_map = get_banned_chunk_map(chunk_db_dataset.chunk_db)

    print_rank_0(" > get dataset map.")
    chunk_dataset_map = get_gpt_chunk_dataset_map()

    # Bert embedder.
    embedder = BertEmbedder(args.retro_bert_max_chunk_length)

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in chunk_dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        # query_dataset_neighbors(index, banned_chunk_map,
        #                         prefix,
        #                         info["embed_dir"], info["nbr_dir"],
        #                         info["data"])
        query_dataset_neighbors(index, banned_chunk_map,
                                prefix, info["data"], info["nbr_dir"],
                                embedder)
# def query_pretraining_neighbors(timer):

#     args = get_args()

#     # Set num threads (torch.distributed reset it to 1).
#     faiss.omp_set_num_threads(8)

#     # Data stuff.
#     pretraining_chunk_dataset_map = get_pretraining_chunk_dataset_map()
#     db_chunk_dataset = get_db_chunk_dataset_map()
#     # text_dataset_map = {key : {
#     #     **info,
#     #     "data" : GPTToTextDataset(info["data"]),
#     # } for key, info in gpt_dataset_map.items()}

#     pax(0, {
#         "pretraining_chunk_dataset_map" : pretraining_chunk_dataset_map,
#         "db_chunk_dataset" : db_chunk_dataset,
#     })
    
#     # Embed.
#     embed_text_datasets(text_dataset_map,
#                         args.retro_bert_max_chunk_length,
#                         args.retro_block_size)
