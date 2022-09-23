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

from functools import reduce
# import glob
import h5py
# import joblib
import json
import numpy as np
import os
from pathlib import Path
# import time
import torch
from tqdm import tqdm

# import sys
# sys.path.append("/home/boxinw-src/megatron-lm/megatron")
# sys.path.append("/home/boxinw-src/megatron-lm/")

# from megatron import get_args
# from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
# from megatron.tokenizer import build_tokenizer

from .utils import (
    get_individual_chunk_index_path,
    get_full_chunk_index_path,
    get_sampled_chunk_index_path,
)

# >>>
from lutil import pax
# <<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# see: notebook/faiss/create_chunks.ipynb
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def get_sorted_dataset_metadatas(args, workdir):

    assert len(args.data_path) % 2 == 0, \
        "currently, only blendable dataset is supported."

    # Data metadata.
    data_metas = []
    for i in range(0, len(args.data_path), 2):
        ratio = float(args.data_path[i])
        prefix = args.data_path[i + 1]
        path = prefix + ".bin"
        name = os.path.basename(prefix)
        assert os.path.exists(path)
        data_metas.append({
            "ratio" : ratio,
            "prefix" : prefix,
            "path" : path,
            "name" : name,
            "chunk_index_path" : get_individual_chunk_index_path(workdir, name)
        })

    # Deterministic dataset order (alphabetical).
    data_metas.sort(key = lambda m : m["prefix"])

    return data_metas

def save_dataset_metadatas(workdir, data_metas):

    # Save dataset order.
    order_path = os.path.join(workdir, "order.json")
    with open(order_path, "w") as f:
        json.dump(data_metas, f)

def build_individual_chunk_index(args, indexed_dataset):

    size = indexed_dataset.sizes.shape[0]
    train = int(round(float(size) * 0.98))

    # eods = []
    chunk_index = []

    for document_id, document in enumerate(tqdm(indexed_dataset)):

        # >>>
        # if document_id == 1000:
        #     break
        # <<<

        if document_id == train:
            break

        eod = document[-1]
        document = document[:-1]
        document_len = len(document)

        chunk_start_idxs = list(range(0, document_len, args.retrieval_chunk_len))
        chunk_end_idxs = [min(document_len, s + args.retrieval_chunk_len)
                          for s in chunk_start_idxs]

        # eods.append(eod)
        chunk_index.extend([(document_id, *idxs)
                            for idxs in zip(chunk_start_idxs, chunk_end_idxs)])

    print(' > converting chunk index to numpy.')
    # eods = np.array(eods)
    chunk_index = np.array(chunk_index)

    # return eods, chunk_index
    return chunk_index

def build_individual_chunk_indexes(args, workdir, data_metas):

    print(" > build individual chunk indexes.")
    for data_index, data_meta in enumerate(data_metas):

        chunk_index_path = data_meta["chunk_index_path"]

        if os.path.exists(chunk_index_path):
            continue

        print(" > building individual chunk index, dataset %d / %d ... '%s'." %
              (data_index, len(data_metas), data_meta["name"]))

        indexed_dataset = make_indexed_dataset(data_meta["prefix"], "mmap", True)
        chunk_index = build_individual_chunk_index(args, indexed_dataset)

        print(" > saving chunk index.")

        f = h5py.File(chunk_index_path, "w")
        # dset = f.create_dataset("eods", data = eods)
        dset = f.create_dataset("chunks", data = chunk_index)
        f.close()

        print(" > finished saving chunk index.")

    # # >>>
    # # rename.
    # for m in data_metas:
    #     print("rename '%s'." % m["name"])
    #     f = h5py.File(m["chunk_index_path"], "w")
    #     f["chunks"] = f["index"]
    #     del f["index"]
    #     f.close()
    # raise Exception("renamed.")
    # # <<<

    # Set n_chunks, n_chunks_sampled (for unambiguity).
    print(" > compute n_chunks, n_chunks_sampled.")
    for data_index, data_meta in enumerate(tqdm(data_metas)):

        f = h5py.File(data_meta["chunk_index_path"], "r")
        data_meta["n_chunks"] = len(f["index"])
        f.close()

        data_meta["n_chunks_sampled"] = \
            int(round(args.retrieval_nchunks_sampled * data_meta["ratio"]))

        pax({"data_meta": data_meta})

def build_full_chunk_index(args, workdir, data_metas):

    # Count total chunks.
    n_chunks = 0
    dtype = None
    for data_index, data_meta in enumerate(data_metas):

        f = h5py.File(data_meta["chunk_index_path"], "r")
        n_chunks += len(f["index"])
        dtype = f["index"].dtype
        f.close()

        print(" > counting chunks, dataset %d / %d, total %d ... '%s'." %
              (data_index, len(data_metas), n_chunks, data_meta["name"]))

    # pax({
    #     "n_chunks" : n_chunks,
    #     "dtype" : dtype,
    # })

    # Full chunk index path.
    full_index_path = get_full_chunk_index_path(workdir)

    # Delete existing chunk index if incorrect size.
    if os.path.exists(full_index_path):

        f = h5py.File(full_index_path)
        n_alloc = len(f["chunks"])           # total allocated
        n_written = f["n_written"][0].item() # total written
        f.close()

        if n_chunks != n_alloc or n_chunks != n_written:
            raise Exception("temporarily disabled.")
            os.remove(full_index_path)

    # Build full chunk index.
    if not os.path.exists(full_index_path):

        f = h5py.File(full_index_path, "w")
        chunk_index = f.create_dataset("chunks", (n_chunks, 3), dtype = dtype)
        dataset_offsets = f.create_dataset(
            "dataset_offsets", (len(data_metas) + 1,), dtype = "uint64")
        n_written = f.create_dataset("n_written", (1,), dtype = "uint64")
        n_written[0] = 0

        start_index = 0
        for data_index, data_meta in enumerate(data_metas):

            print(" > concatenating chunks, dataset %d / %d ... '%s'." %
                  (data_index, len(data_metas), data_meta["name"]))

            g = h5py.File(data_meta["chunk_index_path"], "r")
            data = g["index"]
            chunk_index[start_index:start_index + len(data)] = data
            start_index += len(data)
            dataset_offsets[data_index + 1] = start_index
            n_written[0] = start_index
            g.close()

        f.close()

def build_sampled_chunk_index(args, workdir, data_metas):

    # full_index_path = get_full_chunk_index_path(workdir)
    sampled_index_path = get_sampled_chunk_index_path(workdir)

    # # Count total chunks.
    # f = h5py.File(full_index_path, "r")
    # n_chunks_full = len(f["chunks"])
    # # pax({
    # #     "dataset_offsets" : np.copy(f["dataset_offsets"]),
    # #     "n_written" : np.copy(f["n_written"]),
    # # })
    # f.close()

    # >>>>>>>>  [ **** hack, until using full dataset. **** ] >>>>>>>>
    # Compute sampled chunk count.
    # n_chunks_sampled = reduce(
    #     lambda acc, m : acc + int(round(args.retrieval_nchunks_sampled * m["ratio"])),
    #     data_metas,
    #     0,
    # )
    n_chunks_sampled = sum(m["n_chunks_sampled"] for m in data_metas)
    # <<<<<<<<

    pax({
        "full_index_path" : full_index_path,
        "sampled_index_path" : sampled_index_path,
        "n_chunks_full" : n_chunks_full,
        "n_chunks_sampled / arg" : args.retrieval_nchunks_sampled,
        "n_chunks_sampled" : n_chunks_sampled,
    })

    # Delete existing chunk index if incorrect size.
    if os.path.exists(sampled_index_path):

        raise Exception("sampled index exists.")

        f = h5py.File(sampled_index_path)
        n_alloc = len(f["chunks"])           # total allocated
        n_written = f["n_written"][0].item() # total written
        f.close()

        if n_chunks != n_alloc or n_chunks != n_written:
            raise Exception("temporarily disabled [ sampled ].")
            os.remove(sampled_index_path)

    # Build sampled chunk index.
    if not os.path.exists(sampled_index_path):

        f = h5py.File(chunk_index_path, "w")
        chunk_index = f.create_dataset("chunks", (n_chunks, 3), dtype = dtype)
        dataset_offsets = f.create_dataset(
            "dataset_offsets", (len(data_metas) + 1,), dtype = "uint64")
        n_written = f.create_dataset("n_written", (1,), dtype = "uint64")
        n_written[0] = 0

        start_index = 0
        for data_index, data_meta in enumerate(data_metas):

            print(" > concatenating chunks, dataset %d / %d ... '%s'." %
                  (data_index, len(data_metas), data_meta["name"]))

            g = h5py.File(data_meta["chunk_index_path"], "r")
            data = g["index"]
            # dddn = 
            raise Exception("hi.")
            chunk_index[start_index:start_index + len(data)] = data
            start_index += len(data)
            chunks_written[0] = start_index
            g.close()

        f.close()

# def dump_document_order():
def save_document_order(args, workdir):

    assert torch.distributed.get_rank() == 0, "single process operation."

    # Dataset metadata. (sorted, official order)
    data_metas = get_sorted_dataset_metadatas(args, workdir)
    save_dataset_metadatas(workdir, data_metas)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # create_data_softlinks(data_files)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Build chunk indexes.
    build_individual_chunk_indexes(args, workdir, data_metas)
    build_full_chunk_index(args, workdir, data_metas)
    build_sampled_chunk_index(args, workdir, data_metas)

    raise Exception("finished creating chunks.")

    # joblib.dump(orders, "order.pkl")

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # ['order.pkl']
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # f = h5py.File("sampled_pretraining_corpus" + ".chunks.hdf5", "w")
    # sampled_tot = 300000000
    # dset = f.create_dataset("chunks", (sampled_tot,64), dtype="uint16")

    # pointer = 0
    # for order in tqdm(orders):
    #     dataset = order[0]
    #     ratio = order[1]
    #     size = int(round(float(sampled_tot) * ratio))

    #     rf = h5py.File(dataset, "r")
    #     data = rf["chunks"]
    #     dset[pointer:pointer + size] = data[:size]
    #     pointer += size

    # f.close()

    # f = h5py.File("pretraining_corpus" + ".chunks.hdf5", "r")

    # f['chunks'][2323453]

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # array([  547, 20467, 45427,    13,   632,   561,  1011,  4647,   284,
    # #        30282,   262,  3580,  1022,  3288,   290,  7593,  4808,  7645,
    # #           62, 27997,    13,  1892, 12362,    11,   262,  3288,  4808,
    # #         7645,    62, 27997,   287,  9215,  2900,   503,   284,   307,
    # #        13205,    11,  9472,   262,  7593,   318, 21499,  2728,  2279,
    # #          422,  4890,   284,  2612,  4369,   284, 47906, 15885,   198,
    # #          198,  1135,   783,   760,   326, 23426,   960,  8201,  5384,
    # #          960], dtype=uint16)
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # f['chunks'].shape

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # (5334816766, 64)
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # raise Exception("it worked?")

# eof
