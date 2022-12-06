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
from tools.retro.utils import get_gpt_tokenizer

# >>>
from lutil import pax, print_seq
# <<<


class GPTChunkDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            indexed_datasets,
            chunk_db,
            max_gpt_chunk_length,
    ):

        assert chunk_db.shape[1] == 5, "expected 5 columns (dataset_idx, doc_idx, token_start_idx, token_end_idx, bert_chunk_length); found %d columns." % chunk_db.shape[1]

        self.indexed_datasets = indexed_datasets
        self.chunk_db = chunk_db

        self.max_gpt_chunk_length = max_gpt_chunk_length
        self.gpt_tokenizer = get_gpt_tokenizer()


    def __len__(self):
        return len(self.chunk_db)


    def __getitem__(self, chunk_id):

        indexed_dataset_id, doc_id, token_start_idx, token_end_idx, _ = \
            [ value.item() for value in self.chunk_db[chunk_id] ]
        # pax(0, {
        #     "chunk_db" : self.chunk_db,
        #     "chunk db row" : self.chunk_db[chunk_id],
        #     "indexed_dataset_id" : indexed_dataset_id,
        #     "doc_id" : doc_id,
        #     "token_start_idx" : token_start_idx,
        #     "token_end_idx" : token_end_idx,
        # })
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

        return {
            "doc_id" : doc_id,
            "text" : np.array(token_ids, dtype=np.int64),
        }
