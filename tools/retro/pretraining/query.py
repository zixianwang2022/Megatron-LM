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
from functools import reduce
import glob
import h5py
import numpy as np
import os
import torch
from tqdm import tqdm

from megatron import get_args, mpu, print_rank_0
# from tools.retro.db.dataset import dataset_offsets_to_ids
# from tools.retro.db.utils import get_db_info_map
from tools.retro.index.factory import IndexFactory
from tools.retro.index.utils import get_index_workdir

from .chunk_dataset import get_gpt_chunk_dataset_map

# >>>
from lutil import pax, print_seq, shorten as shorten_str
# <<<


def get_index():

    args = get_args()

    # Load index.
    index_wrapper = IndexFactory.get_index(args.retro_index_ty)
    index_dir = get_index_workdir()
    added_index_path = index_wrapper.get_added_index_path(None, index_dir)
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


def get_banned_chunk_map():

    # Load chunk db.
    print_rank_0("load chunk db.")
    chunk_db_path = get_db_info_map()["full"]["db_path"]
    f = h5py.File(chunk_db_path, "r")
    doc_ids = np.copy(f["chunks_valid"][:, 0]).tolist()
    dataset_offsets = np.copy(f["dataset_offsets_valid"]).tolist()
    f.close()

    dataset_ids = dataset_offsets_to_ids(dataset_offsets)
    assert len(doc_ids) == len(dataset_ids)

    # >>>
    # pax(0, {
    #     "doc_ids" : "%d / %s" % (len(doc_ids), str(doc_ids)),
    #     "dataset_offsets": "%d / %s"%(len(dataset_offsets), str(dataset_offsets)),
    #     "dataset_ids" : "%d / %s" % (len(dataset_ids), str(dataset_ids)),
    #     "doc_ids / 836728" : [
    #         (dataset_ids[i], doc_id)
    #         for i, doc_id in enumerate(doc_ids)
    #         if doc_id == 836728
    #     ],
    # })
    # <<<

    # Map docs to chunks.
    print_rank_0("build doc-chunk-id-map.")
    banned_chunk_map = defaultdict(set)
    for chunk_id, doc_id in enumerate(doc_ids):
        if chunk_id % 10000000 == 0:
            print_rank_0("mapping banned chunks, %.0f%%." %
                         (100 * chunk_id / len(doc_ids)))
        dataset_id = dataset_ids[chunk_id]
        banned_chunk_map[(dataset_id, doc_id)].add(chunk_id)

    # >>>
    # n_empties = len([v for v in banned_chunk_map.values() if not v])
    # pax(0, {
    #     "chunk_db_path" : chunk_db_path,
    #     "banned_chunk_map" : {
    #         d : "%d / %s" % (len(cs), shorten_str(str(cs), 50))
    #         for i, (d, cs) in enumerate(banned_chunk_map.items())
    #         if i < 10 or i >= len(banned_chunk_map) - 10
    #     },
    #     "n_empties" : n_empties,
    # })
    # <<<

    return banned_chunk_map


def get_missing_neighbor_blocks(embed_dir, nbr_dir):

    # Delete corrupt files.
    print_rank_0("delete corrupt neighbor blocks.")
    if torch.distributed.get_rank() == 0:
        existing_block_paths = glob.glob(nbr_dir + "/*.hdf5")
        pbar = tqdm(existing_block_paths)
        for index, path in enumerate(pbar):
            pbar.set_description("verifying neighbor block.")
            f = h5py.File(path, "r")
            try:
                assert len(f["neighbors"].shape) == 2
                f.close()
            except Exception as e:
                # >>>
                raise Exception("corrupt neighbors?")
                # <<<
                os.remove(path)

    # Wait for files to be deleted.
    torch.distributed.barrier()

    # Missing files.
    print_rank_0("all missing neighbor blocks.")
    all_files = set(os.path.basename(f)
                    for f in glob.glob(embed_dir + "/*.hdf5"))
    existing_files = set(os.path.basename(f)
                         for f in glob.glob(nbr_dir + "/*.hdf5"))
    missing_files = sorted(list(all_files - existing_files))

    # pax(0, {
    #     "embed_dir" : embed_dir,
    #     "nbr_dir" : nbr_dir,
    #     "all_files" : all_files,
    #     "existing_files" : existing_files,
    #     "missing_files" : missing_files,
    # })

    # This rank's missing files.
    print_rank_0("rank's missing neighbor blocks.")
    data_parallel_rank = mpu.get_data_parallel_rank()
    data_parallel_world_size = mpu.get_data_parallel_world_size()
    rank_missing_files = missing_files[data_parallel_rank:len(missing_files):data_parallel_world_size]
    rank_missing_blocks = [{
        "range" : tuple(int(i) for i in os.path.splitext(f)[0].split("-")),
        "embed_path" : os.path.join(embed_dir, f),
        "nbr_path" : os.path.join(nbr_dir, f),
    } for f in rank_missing_files]

    # pax(0, {
    #     "rank_missing_blocks" : rank_missing_blocks,
    #     "rank_missing_blocks / 0" : rank_missing_blocks[0],
    # })

    # Extend rank's missing items (with None) such that all ranks have equal
    # length lists. This allows for easier tracking of global progress.
    print_rank_0("synchronize missing block count.")
    n_missing_tensor = torch.cuda.LongTensor([len(rank_missing_blocks)])
    torch.distributed.all_reduce(n_missing_tensor,
                                 op = torch.distributed.ReduceOp.MAX)
    max_n_missing = n_missing_tensor.item()
    rank_missing_blocks += [None] * (max_n_missing - len(rank_missing_blocks))

    # >>>
    # print_seq("missing blocks [%d] : %s ... %s." % (
    #     len(rank_missing_blocks),
    #     str(rank_missing_blocks[0]["range"]),
    #     str(rank_missing_blocks[-1]["range"]) if rank_missing_blocks[-1] else str(rank_missing_blocks[-2]["range"]),
    # ))
    # <<<

    return rank_missing_blocks


def query_dataset_neighbors(index, banned_chunk_map,
                            prefix, embed_dir, nbr_dir, dataset):

    args = get_args()

    missing_nbr_blocks = get_missing_neighbor_blocks(embed_dir, nbr_dir)

    for block_index, block in enumerate(missing_nbr_blocks):

        if block is not None:

            # Progress.
            print_rank_0("query '%s' block %d / %d ... %s." % (
                prefix,
                block_index,
                len(missing_nbr_blocks),
                block["nbr_path"],
            ))

            # Load embeddings.
            print_rank_0("load embeddings.")
            f = h5py.File(block["embed_path"], "r")
            data = np.copy(f["data"])
            f.close()

            # Query neighbor ids.
            print_rank_0("search.")
            _, query_nbr_ids = index.search(data, args.retro_nnbrs_query)

            # Banned neighbor ids.
            print_rank_0("get banned neighbor ids.")
            sample_ids = sorted(list(set(chunk_id // dataset.n_chunks_per_seq
                                         for chunk_id in range(*block["range"]))))
            sample_banned_chunk_id_map = {}
            for sample_id in sample_ids:
                sample = dataset.seq_dataset[sample_id]
                dataset_idx = sample["dataset_idx"].item()
                # doc_ids = set(sample["doc_ids"])
                doc_ids = sample["doc_ids"].tolist()
                banned_chunk_ids = set()
                for doc_id in doc_ids:
                    # >>>
                    # banned_chunk_ids.update(banned_doc_chunk_id_map[doc_id])

                    current_chunk_ids = banned_chunk_map[(dataset_idx, doc_id)]
                    # >>>
                    # if not current_chunk_ids:
                    #     pax(0, {
                    #         "sample_id" : sample_id,
                    #         "sample" : sample,
                    #         "dataset_idx" : dataset_idx,
                    #         "doc_id" : doc_id,
                    #         "current_chunk_ids" : current_chunk_ids,
                    #     })
                    assert current_chunk_ids, "should be >=1 chunk_id."
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
            os.makedirs(os.path.dirname(block["nbr_path"]), exist_ok = True)
            f = h5py.File(block["nbr_path"], "w")
            f.create_dataset("neighbors", data = filtered_nbr_ids)
            f.close()

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()


def query_pretraining_neighbors(timer):

    # >>>
    # Set num threads (torch.distributed reset it to 1).
    # assert torch.distributed.get_rank() == 0
    # if torch.distributed.get_rank() != 0:
    #     return
    # faiss.omp_set_num_threads(64)
    faiss.omp_set_num_threads(8)
    # <<<

    # Load index, banned chunk ids, datasets.
    print_rank_0(" > get index.")
    index = get_index()

    print_rank_0(" > get banned doc-chunk id map.")
    banned_chunk_map = get_banned_chunk_map()

    print_rank_0(" > get dataset map.")
    chunk_dataset_map = get_gpt_chunk_dataset_map()

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in chunk_dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        query_dataset_neighbors(index, banned_chunk_map,
                                prefix,
                                info["embed_dir"], info["nbr_dir"],
                                info["data"])
