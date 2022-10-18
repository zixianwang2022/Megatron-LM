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
from collections import defaultdict
import faiss
# from faiss import ParameterSpace
from functools import reduce
import glob
import h5py
# import joblib
# import multiprocessing
import numpy as np
import os
# import psutil
# import time
import torch
from tqdm import tqdm

from megatron import mpu, print_rank_0
from tools.retro.db.utils import get_full_db_info
from tools.retro.index.factory import IndexFactory
from tools.retro.index.utils import get_index_workdir

from .dataset import get_dataset_map

# >>>
from lutil import pax, print_seq
# <<<


def get_index(args):

    # Load index.
    index_wrapper = IndexFactory.get_index(args)
    index_dir = get_index_workdir(args)
    added_index_path = index_wrapper.get_added_index_path(None, index_dir)
    index = faiss.read_index(added_index_path)

    # Search parameters.
    raise Exception("update search params.")
    if 0:
        ParameterSpace().set_index_parameter(index, "efSearch", args.efsearch)
        ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)

    # pax(0, {
    #     "index_wrapper" : index_wrapper,
    #     "index" : index,
    #     "index_dir" : index_dir,
    #     "added_index_path" : added_index_path,
    # })

    return index


# def get_chunk_db(args):
def get_banned_doc_chunk_id_map(args):

    chunk_db_path = get_full_db_info(args)["db_path"]
    f = h5py.File(chunk_db_path, "r")
    doc_ids = np.copy(f["chunks_valid"][:, 0])
    f.close()

    doc_chunk_id_map = defaultdict(set)
    for chunk_id, doc_id in enumerate(doc_ids):
        doc_chunk_id_map[doc_id].add(chunk_id)

    # pax(0, {
    #     "chunk_db_path" : chunk_db_path,
    #     "doc_chunk_id_map" : {
    #         d : "%d / %s" % (len(cs), str(cs))
    #         for d, cs in doc_chunk_id_map.items()
    #         if d < 20
    #     },
    # })

    return doc_chunk_id_map


# def get_missing_neighbor_blocks(args, workdir, dataset_key):
def get_missing_neighbor_blocks(embed_dir, nbr_dir):

    # Delete corrupt files.
    if torch.distributed.get_rank() == 0:
        existing_block_paths = glob.glob(nbr_dir + "/*.hdf5")
        pbar = tqdm(existing_block_paths)
        for index, path in enumerate(pbar):
            pbar.set_description("verifying neighbor block.")
            f = h5py.File(path, "r")
            try:
                assert len(f["data"].shape) == 2
            except:
                raise Exception("delete block file.")
            finally:
                f.close()

    # Wait for files to be deleted.
    torch.distributed.barrier()

    # Missing files.
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


# def query_neighbors_single(args, workdir, dataset_key, dataset_path):
def query_dataset_neighbors(args, workdir, index, banned_doc_chunk_id_map,
                            prefix, embed_dir, nbr_dir, dataset):

    missing_nbr_blocks = get_missing_neighbor_blocks(embed_dir, nbr_dir)

    # pax(0, {
    #     "workdir" : workdir,
    #     "index" : index,
    #     "prefix" : prefix,
    #     "embed_dir" : embed_dir,
    #     "nbr_dir" : nbr_dir,
    #     "dataset" : dataset,
    #     "missing_nbr_blocks" : missing_nbr_blocks,
    #     "missing_nbr_blocks / 0" : missing_nbr_blocks[0],
    # })

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
            f = h5py.File(block["embed_path"], "r")
            data = np.copy(f["data"])
            f.close()

            # Query neighbor ids.
            _, query_nbr_ids = index.search(data, args.retro_nnbrs_query)

            # Banned neighbor ids.
            sample_ids = sorted(list(set(chunk_id // dataset.n_chunk_seq_ratio
                                         for chunk_id in range(*block["range"]))))
            sample_banned_chunk_id_map = {}
            for sample_id in sample_ids:
                banned_doc_ids = set(dataset.seq_dataset[sample_id]["doc_ids"])
                banned_chunk_ids = set()
                for doc_id in banned_doc_ids:
                    banned_chunk_ids.update(banned_doc_chunk_id_map[doc_id])
                sample_banned_chunk_id_map[sample_id] = banned_chunk_ids

            # Filter banned neighbor ids.
            filtered_nbr_ids = np.full(
                shape = (len(query_nbr_ids), args.retro_nnbrs_target),
                fill_value = -1,
                dtype = "int64",
            )
            min_chunk_id, max_chunk_id = block["range"]
            for chunk_id in range(min_chunk_id, max_chunk_id):
                sample_id = chunk_id // dataset.n_chunk_seq_ratio
                query_row = [i for i in query_nbr_ids[chunk_id-min_chunk_id] if i>=0]
                filtered_row = [i for i in query_row
                             if i not in sample_banned_chunk_id_map[sample_id]]
                filtered_row = filtered_row[:200]
                filtered_row += [-1] * (args.retro_nnbrs_target - len(filtered_row))
                filtered_nbr_ids[chunk_id-min_chunk_id] = filtered_row

            # pax(0, {
            #     # "data" : data,
            #     # "query_nbr_ids" : query_nbr_ids,
            #     "dataset" : dataset,
            #     "sample_ids" : "%d / %s" % (len(sample_ids), str(sample_ids)),
            #     "sample_banned_chunk_id_map" : sample_banned_chunk_id_map,
            #     "filtered_nbr_ids" : filtered_nbr_ids.tolist(),
            # })

            os.makedirs(os.path.dirname(block["nbr_path"]), exist_ok = True)
            f = h5py.File(block["nbr_path"], "w")
            f.create_dataset("neighbors", data = filtered_nbr_ids)
            f.close()

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()


def query_neighbors(args, workdir, timer):

    # >>>
    # # Set num threads (torch.distributed reset it to 1).
    # # assert torch.distributed.get_rank() == 0
    # if torch.distributed.get_rank() != 0:
    #     return
    # faiss.omp_set_num_threads(64)
    # <<<

    # Load index, banned chunk ids, datasets.
    print_rank_0(" > get index.")
    index = get_index(args)

    print_rank_0(" > get banned doc-chunk id map.")
    banned_doc_chunk_id_map = get_banned_doc_chunk_id_map(args)

    print_rank_0(" > get dataset map.")
    dataset_map = get_dataset_map(args, workdir)

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        query_dataset_neighbors(args, workdir, index, banned_doc_chunk_id_map,
                                prefix,
                                info["embed_dir"], info["nbr_dir"],
                                info["data"])

