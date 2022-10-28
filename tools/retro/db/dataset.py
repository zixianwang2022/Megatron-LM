# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import json
import numpy as np
import torch

from megatron import get_args, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from tools.retro.utils import get_gpt_tokenizer

from .utils import get_db_info_map, get_indexed_dataset_infos

# >>>
from lutil import pax, print_seq
# <<<


class GPTChunkDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            indexed_datasets,
            indexed_dataset_ids,
            chunk_index,
            max_gpt_chunk_length,
    ):

        self.indexed_datasets = indexed_datasets
        self.indexed_dataset_ids = indexed_dataset_ids
        self.chunk_index = chunk_index

        self.max_gpt_chunk_length = max_gpt_chunk_length
        self.gpt_tokenizer = get_gpt_tokenizer()


    def __len__(self):
        return len(self.chunk_index)


    def __getitem__(self, chunk_id):

        indexed_dataset_id = self.indexed_dataset_ids[chunk_id]
        doc_id, token_start_idx, token_end_idx, _ = \
            [ value.item() for value in self.chunk_index[chunk_id] ]
        chunk_length = token_end_idx - token_start_idx
        indexed_dataset = self.indexed_datasets[indexed_dataset_id]

        token_ids = indexed_dataset.get(doc_id,
                                        offset = token_start_idx,
                                        length = chunk_length)

        if chunk_length != self.max_gpt_chunk_length:
            assert chunk_length < self.max_gpt_chunk_length, "invalid chunk len."
            token_ids = token_ids.tolist()
            token_ids += [self.gpt_tokenizer.eod_id] * \
                (self.max_gpt_chunk_length - chunk_length)

        # pax({
        #     # "indexed_dataset" : indexed_dataset,
        #     "chunk_id" : chunk_id,
        #     "dataset_id" : dataset_id,
        #     "doc_id" : doc_id,
        #     "token_start_idx" : token_start_idx,
        #     "token_end_idx" : token_end_idx,
        #     "chunk" : chunk,
        # })

        return {'text': np.array(token_ids, dtype=np.int64)}


def dataset_offsets_to_ids(offsets):
    ids = []
    for i in range(len(offsets) - 1):
        ids.append([i] * (offsets[i+1] - offsets[i]))
    ids = [ i for ii in ids for i in ii ]
    # pax(0, {"offsets": str(offsets), "ids": str(ids)})
    return ids


def get_gpt_chunk_dataset_map():

    args = get_args()

    # Load indexed dataset infos.
    indexed_dataset_infos = get_indexed_dataset_infos()

    # Indexed datasets.
    indexed_datasets = []
    for index, indexed_dataset_info in enumerate(indexed_dataset_infos):
        print_rank_0("indexed dataset %d / %d [ %s ]." % (
            index,
            len(indexed_dataset_infos),
            indexed_dataset_info["prefix"],
        ))
        indexed_datasets.append(
            make_indexed_dataset(indexed_dataset_info["prefix"], "mmap", True))

    # Chunk index.
    db_info_map = get_db_info_map()
    chunk_dataset_map = {}
    for db_index, (db_key, db_info) in enumerate(db_info_map.items()):

        print_rank_0("init gpt chunk dataset %d / %d [ %s ]." %
              (db_index, len(db_info_map), db_key))

        # Load chunk index.
        f = h5py.File(db_info["db_path"], "r")
        indexed_dataset_offsets = np.copy(f["dataset_offsets_valid"])
        chunk_index = np.copy(f["chunks_valid"])
        f.close()

        # Indexed dataset ids.
        indexed_dataset_ids = dataset_offsets_to_ids(indexed_dataset_offsets)

        # Chunk dataset.
        chunk_dataset_map[db_key] = {
            "data" : GPTChunkDataset(
                indexed_datasets = indexed_datasets,
                indexed_dataset_ids = indexed_dataset_ids,
                chunk_index = chunk_index,
                max_gpt_chunk_length = args.retro_gpt_chunk_length,
            ),
            "embed_dir" : db_info["embed_dir"],
        }

    return chunk_dataset_map
