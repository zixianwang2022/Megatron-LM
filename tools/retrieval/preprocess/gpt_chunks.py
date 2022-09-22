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

import h5py
import numpy as np
import os
from pathlib import Path
import time
import torch
from tqdm import tqdm

# from megatron import get_args
# from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
# from megatron.training import (
#     build_train_valid_test_data_iterators,
#     update_train_iters,
# )
# from pretrain_gpt import train_valid_test_datasets_provider

# >>>
from lutil import pax
# <<<


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    print(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


# def get_database_and_index(args, indexed_dataset):
def get_chunk_index(args, indexed_dataset):

    size = indexed_dataset.sizes.shape[0]
    train = int(round(float(size) * 0.98))
    # tot = 0

    # databases = []
    # indexes = []
    eods = []
    # chunk_ranges = []
    chunk_index = []

    # pax({"num docs": len(indexed_dataset.doc_idx)})

    for document_id, document in enumerate(tqdm(indexed_dataset)):

        # >>>
        # if document_id == 10:
        #     break
        # <<<
        
        if document_id == train:
            break

        eod = document[-1]
        document = document[:-1]
        document_len = len(document)

        chunk_start_idxs = list(range(0, document_len, args.retriever_chunk_len))
        chunk_end_idxs = [min(document_len, s + args.retriever_chunk_len)
                          for s in chunk_start_idxs]
        # chunk_ranges = list(zip(chunk_start_idxs, chunk_end_idxs))

        eods.append(eod)
        # chunk_ranges.append(list(zip(chunk_start_idxs, chunk_end_idxs)))
        chunk_index.extend([(document_id, *idxs)
                            for idxs in zip(chunk_start_idxs, chunk_end_idxs)])

        # pax({
        #     # "chunk_start_idxs" : chunk_start_idxs,
        #     # "chunk_end_idxs" : chunk_end_idxs,
        #     "chunk_ranges" : chunk_ranges,
        # })

        # token_no = len(document)
        # # tot += token_no
        # chunks = int(np.ceil(token_no / args.retriever_chunk_len))

        # # >>>
        # pax({
        #     "document_id" : document_id,
        #     "document" : document,
        #     "eod" : eod,
        #     "token_no" : token_no,
        #     "chunks" : chunks,
        # })
        # # <<<

        # for i in range(chunks):
        #     tokens = document[i * 64:(i+1) *64]
        #     if len(tokens) < 64:
        #         pad = np.array([eod] * (64 - len(tokens)), dtype='uint16')
        #         tokens = np.hstack((tokens, pad))
        #     assert len(tokens) == 64
        #     databases.append(tokens)
        #     indexes.append(document_id)

    # raise Exception("hi.")
    # pax({
    #     # "eods" : eods,
    #     "eods / np" : np.array(eods),
    #     # "chunk_ranges" : chunk_ranges,
    #     # "chunk_ranges / np" : np.array(chunk_ranges),
    #     # "chunk_index" : chunk_index,
    #     "chunk_index / np" : np.array(chunk_index),
    # })

    print(' > converting chunk index to numpy.')

    # return databases, indexes
    return np.array(eods), np.array(chunk_index)


def build_gpt_chunk_index(args, timer):

    assert torch.distributed.get_rank() == 0, "single process operation."

    data_files = [ prefix.rstrip("/") + ".bin" for prefix in args.data_path ]
    data_files = [ path for path in data_files if os.path.exists(path) ]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if 0:
        create_data_softlinks(data_files)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # pax(0, {"data_files": data_files})

    for data_index, data_file in enumerate(data_files):

        data_name = Path(data_file).stem
        data_prefix = os.path.splitext(data_file)[0]
        chunk_index_file = \
            data_prefix + f".chunk_index_n{args.retriever_chunk_len}.hdf5"

        # >>>
        # pax(0, {
        #     "data_file" : data_file,
        #     "chunk_file" : chunk_file,
        # })
        # <<<

        if os.path.exists(chunk_index_file):
            # raise Exception("chunk index exists.")
            continue

        # >>>
        # pax(0, {"data_name": data_name, "data_prefix": data_prefix})
        # print(data_name)
        # <<<

        print(" > creating chunk index, dataset %d / %d ... '%s'." %
              (data_index, len(data_files), data_name))

        indexed_dataset = get_indexed_dataset_(data_prefix, "mmap", True)
        # databases, indexes = get_database_and_index(args, indexed_dataset)
        eods, chunk_index = get_chunk_index(args, indexed_dataset)

        # pax({
        #     "indexed_dataset" : indexed_dataset,
        #     "databases" : databases,
        #     "indexes" : indexes,
        # })
        # pax({
        #     "eods" : eods.shape,
        #     "chunk_index" : chunk_index,
        # })

        # database = np.vstack(databases)
        # index = np.array(indexes)

        print(" > saving chunk index.")

        f = h5py.File(chunk_index_file, "w")
        # dset = f.create_dataset("chunks", data=database)
        # dset = f.create_dataset("document_id", data=index)
        dset = f.create_dataset("eods", data=eods)
        dset = f.create_dataset("index", data=chunk_index)
        f.close()

        print(" > finished saving chunk index.")

        # raise Exception("saved '%s'." % chunk_index_file)
        # pax(0, {
        #     "data_file" : data_file,
        #     "indexed_dataset" : type(indexed_dataset).__name__,
        #     "database" : type(database).__name__,
        #     "index" : type(index).__name__,
        # })

    # raise Exception("finished creating chunks.")
