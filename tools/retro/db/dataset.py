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

# import numpy as np
import torch

# from megatron import get_args, get_tokenizer, print_rank_0
# from megatron.data.bert_dataset import build_training_sample
from megatron.tokenizer.tokenizer import (
    # _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)

# >>>
from lutil import pax, print_seq
# <<<


class GPTChunkDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            indexed_datasets,
            dataset_ids,
            chunk_index,
            max_chunk_length,
    ):

        self.indexed_datasets = indexed_datasets
        self.dataset_ids = dataset_ids
        self.chunk_index = chunk_index
        self.max_gpt_chunk_length = max_chunk_length

        # >>>
        self.gpt_tokenizer = _GPT2BPETokenizer(
            vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json",
            merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt",
        )
        # <<<


    def __len__(self):
        return len(self.chunk_index)


    def __getitem__(self, chunk_id):

        dataset_id = self.dataset_ids[chunk_id]
        doc_id, token_start_idx, token_end_idx, _ = \
            [ value.item() for value in self.chunk_index[chunk_id] ]
        chunk_length = token_end_idx - token_start_idx
        indexed_dataset = self.indexed_datasets[dataset_id]

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


# def get_dataset_map(args):
def get_gpt_chunk_dataset_map(args):

    # Load dataset metadata.
    data_metas_path = get_dataset_metas_path(args)
    with open(data_metas_path) as f:
        data_metas = json.load(f)

    # Token datasets.
    indexed_datasets = []
    for index, data_meta in enumerate(data_metas):
        print("indexed dataset %d / %d [ %s ]." %
              (index, len(data_metas), data_meta["prefix"]))
        indexed_datasets.append(
            make_indexed_dataset(data_meta["prefix"], "mmap", True))

    # Chunk index.
    db_info_map = get_db_info_map(args)
    dataset_map = {}
    for db_index, (db_key, db_info) in enumerate(db_info_map.items()):

        print("init gpt chunk dataset %d / %d [ %s ]." %
              (db_index, len(db_info_map), db_key))

        # Load chunk index.
        f = h5py.File(db_info["db_path"], "r")
        dataset_offsets = np.copy(f["dataset_offsets_valid"])
        chunk_index = np.copy(f["chunks_valid"])
        f.close()

        # Dataset ids.
        dataset_ids = []
        for i in range(len(dataset_offsets) - 1):
            dataset_ids.append([i] * (dataset_offsets[i+1] - dataset_offsets[i]))
        dataset_ids = [ i for ii in dataset_ids for i in ii ]

        # Dataset.
        dataset_map[db_key] = GPTChunkDataset(
            indexed_datasets = indexed_datasets,
            dataset_ids = dataset_ids,
            chunk_index = chunk_index,
            max_chunk_length = args.retro_chunk_length,
        )

    return dataset_map

