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

# from megatron import get_args
# from megatron import get_args, get_tokenizer, print_rank_0
from megatron import print_rank_0
# from megatron.data.bert_dataset import build_training_sample
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

        # args = get_args()

        self.indexed_datasets = indexed_datasets
        self.indexed_dataset_ids = indexed_dataset_ids
        self.chunk_index = chunk_index

        # self.max_gpt_chunk_length = args.retro_chunk_length
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


class GPTToTextDataset(torch.utils.data.Dataset):


    def __init__(self, gpt_dataset):

        super().__init__()

        self.gpt_dataset = gpt_dataset
        self.gpt_tokenizer = get_gpt_tokenizer()


    def __len__(self):
        return len(self.gpt_dataset)


    def __getitem__(self, idx):
        gpt_token_ids = self.gpt_dataset[idx]["text"].tolist()
        text = self.gpt_tokenizer.detokenize(gpt_token_ids)
        return {"text": text}


# def get_dataset_map(args):
def get_gpt_chunk_dataset_map(args):

    # Load indexed dataset infos.
    indexed_dataset_infos = get_indexed_dataset_infos(args)

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
    db_info_map = get_db_info_map(args)
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
        indexed_dataset_ids = []
        for i in range(len(indexed_dataset_offsets) - 1):
            indexed_dataset_ids.append(
                [i] * (indexed_dataset_offsets[i+1] - indexed_dataset_offsets[i]))
        indexed_dataset_ids = [ i for ii in indexed_dataset_ids for i in ii ]

        # Chunk dataset.
        chunk_dataset_map[db_key] = {
            "data" : GPTChunkDataset(
                indexed_datasets = indexed_datasets,
                indexed_dataset_ids = indexed_dataset_ids,
                chunk_index = chunk_index,
                max_gpt_chunk_length = args.retro_chunk_length,
            ),
            "embed_dir" : db_info["embed_dir"],
        }

    return chunk_dataset_map

