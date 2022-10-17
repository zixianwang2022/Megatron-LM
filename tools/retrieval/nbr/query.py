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
from tools.retrieval.chunks.utils import get_full_chunk_db_path
from tools.retrieval.index.factory import IndexFactory
from tools.retrieval.index.utils import get_index_workdir

from .dataset import get_dataset_map

# >>>
from lutil import pax, print_seq
# <<<


def get_chunk_db(args):

    chunk_db_workdir = os.path.join(args.retro_workdir, "chunks")
    chunk_db_path = get_full_chunk_db_path(chunk_db_workdir)

    pax(0, {
        "chunk_db_workdir" : chunk_db_workdir,
        "chunk_db_path" : chunk_db_path,
    })

def get_index(args):

    # Load index.
    index_wrapper = IndexFactory.get_index(args)
    index_dir = get_index_workdir(args)
    added_index_path = index_wrapper.get_added_index_path(None, index_dir)
    index = faiss.read_index(added_index_path)

    # Search parameters.
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
def query_dataset_neighbors(args, workdir, index, prefix,
                            embed_dir, nbr_dir, dataset):

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

        # Load embeddings.
        f = h5py.File(block["embed_path"], "r")
        data = np.copy(f["data"])
        f.close()

        # Query index.
        _, query_nbr_ids = index.search(data, args.retro_nnbrs_query)
        
        # Banned nbr ids.
        sample_ids = sorted(list(set(chunk_id // dataset.n_chunk_seq_ratio
                                     for chunk_id in range(*block["range"]))))
        # pax(0, {
        #     "chunk range" : block["range"],
        #     "sample_ids" : "%d / %s" % (len(sample_ids), str(sample_ids)),
        # })
        # for chunk_id in range(*block["range"]):
        #     sample_id = chunk_id // dataset.n_chunk_seq_ratio
        #     banned_doc_ids.append(set(dataset[sample_id]["doc_ids"]))
        sample_banned_doc_id_map = {}
        for sample_id in sample_ids:
            sample_banned_doc_id_map[sample_id] = \
                set(dataset.seq_dataset[sample_id]["doc_ids"])
        

        pax(0, {
            # "data" : data,
            # "query_nbr_ids" : query_nbr_ids,
            "dataset" : dataset,
            "sample_ids" : "%d / %s" % (len(sample_ids), str(sample_ids)),
            "sample_banned_doc_id_map" : sample_banned_doc_id_map,
        })

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

        fout = h5py.File(out_path, "w")
        fout.create_dataset("neighbors", data=neighbors)
        fout.close()


def query_neighbors(args, workdir, timer):

    # >>>
    # # Set num threads (torch.distributed reset it to 1).
    # # assert torch.distributed.get_rank() == 0
    # if torch.distributed.get_rank() != 0:
    #     return
    # faiss.omp_set_num_threads(64)
    # <<<

    # Load chunk db, index.
    chunk_db = get_chunk_db(args)
    index = get_index(args)

    # Query each (i.e., train, valid, test) dataset.
    dataset_map = get_dataset_map(args, workdir)
    for prefix, info in dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        query_dataset_neighbors(args, workdir, index, prefix,
                                info["embed_dir"], info["nbr_dir"],
                                info["data"])

