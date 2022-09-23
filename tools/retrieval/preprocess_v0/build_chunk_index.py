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

from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

from .utils import get_chunk_index_path

# >>>
from lutil import pax
# <<<


# def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
#     """Build indexed dataset."""
#     print(' > building dataset index ...')

#     start_time = time.time()
#     indexed_dataset = make_indexed_dataset(data_prefix,
#                                            data_impl,
#                                            skip_warmup)
#     print(' > finished creating indexed dataset in {:4f} '
#                  'seconds'.format(time.time() - start_time))
#     print('    number of documents: {}'.format(
#         indexed_dataset.sizes.shape[0]))

#     return indexed_dataset


# def get_chunk_index(args, indexed_dataset):
def build_single_chunk_index(args, indexed_dataset):

    size = indexed_dataset.sizes.shape[0]
    train = int(round(float(size) * 0.98))

    eods = []
    chunk_index = []

    for document_id, document in enumerate(tqdm(indexed_dataset)):

        if document_id == train:
            break

        eod = document[-1]
        document = document[:-1]
        document_len = len(document)

        chunk_start_idxs = list(range(0, document_len, args.retriever_chunk_len))
        chunk_end_idxs = [min(document_len, s + args.retriever_chunk_len)
                          for s in chunk_start_idxs]

        eods.append(eod)
        chunk_index.extend([(document_id, *idxs)
                            for idxs in zip(chunk_start_idxs, chunk_end_idxs)])

    print(' > converting chunk index to numpy.')
    eods = np.array(eods)
    chunk_index = np.array(chunk_index)

    return eods, chunk_index


# def build_gpt_chunk_index(args, timer):
# def build_chunk_indexes(args, timer):
def build_chunk_index(args, timer):

    assert torch.distributed.get_rank() == 0, "single process operation."

    data_files = [ prefix.rstrip("/") + ".bin" for prefix in args.data_path ]
    data_files = [ path for path in data_files if os.path.exists(path) ]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if 0:
        create_data_softlinks(data_files)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    for data_index, data_file in enumerate(data_files):

        data_name = Path(data_file).stem
        data_prefix = os.path.splitext(data_file)[0]
        chunk_index_file = get_chunk_index_path(args, data_prefix)

        if os.path.exists(chunk_index_file):
            continue

        print(" > creating chunk index, dataset %d / %d ... '%s'." %
              (data_index, len(data_files), data_name))

        # indexed_dataset = get_indexed_dataset_(data_prefix, "mmap", True)
        indexed_dataset = make_indexed_dataset(data_prefix, "mmap", True)
        eods, chunk_index = build_single_chunk_index(args, indexed_dataset)

        print(" > saving chunk index.")

        f = h5py.File(chunk_index_file, "w")
        dset = f.create_dataset("eods", data=eods)
        dset = f.create_dataset("index", data=chunk_index)
        f.close()

        print(" > finished saving chunk index.")
