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

import os
import torch

from megatron import get_retro_args, print_rank_0
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.training import (
    build_train_valid_test_data_loaders,
    update_train_iters,
)
from tools.retro.db.utils import get_indexed_dataset_infos
from tools.retro.utils import get_num_chunks_per_seq

from .utils import get_base_pretraining_workdir

# >>>
from lutil import pax
# <<<


class GPTChunkDataset(torch.utils.data.Dataset):

    def __init__(self, seq_dataset, chunk_length):

        super().__init__()

        self.seq_dataset = seq_dataset

        self.chunk_length = chunk_length
        self.n_chunks_per_seq = get_num_chunks_per_seq()
        self.n_seqs = len(seq_dataset)
        self.n_chunks = self.n_seqs * self.n_chunks_per_seq


    def __len__(self):
        return self.n_chunks


    def __getitem__(self, idx):

        # >>>
        # seq_idx = idx // self.n_chunk_seq_ratio
        # chunk_idx = idx % self.n_chunk_seq_ratio
        seq_idx = idx // self.n_chunks_per_seq
        chunk_idx = idx % self.n_chunks_per_seq
        # <<<

        seq_sample = self.seq_dataset[seq_idx]
        seq_token_ids = seq_sample["text"]
        seq_doc_ids = seq_sample["doc_ids"]

        # assert len(seq_token_ids) == self.seq_length, \
        #     "len(seq_token_ids) == %d." % len(seq_token_ids)

        token_start_idx = chunk_idx * self.chunk_length
        token_end_idx = token_start_idx + self.chunk_length
        chunk_token_ids = seq_token_ids[token_start_idx:token_end_idx]

        return {
            "text" : chunk_token_ids,
            "doc_ids" : seq_doc_ids,
        }


def verify_indexed_dataset_order():

    args = get_retro_args()

    db_indexed_dataset_infos = get_indexed_dataset_infos()
    db_prefixes = [ info["prefix"] for info in db_indexed_dataset_infos ]

    assert len(args.data_path) >= 2, "blendable dataset supported only."
    pretraining_prefixes = args.data_path[1:None:2]

    if len(db_prefixes) != len(pretraining_prefixes):
        raise Exception("inconsistent dataset count between db & pretraining.")
    if db_prefixes != pretraining_prefixes:
        raise Exception("inconsistent dataset order between db & pretraining.")

    # pax(0, {
    #     "db_prefixes" : db_prefixes,
    #     "pretraining_prefixes" : pretraining_prefixes,
    # })


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    args = get_retro_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.retro_gpt_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        return_doc_ids=args.retro_return_doc_ids)
    print_rank_0("> finished creating pretrained GPT datasets ...")

    # >>>
    # pax(0, {
    #     "train_ds" : len(train_ds),
    #     "valid_ds" : len(valid_ds),
    #     # "test_ds" : len(test_ds),
    # })
    # <<<

    return train_ds, valid_ds, test_ds


def get_gpt_chunk_dataset_map():

    args = get_retro_args()

    # Update train iters.
    update_train_iters(args)

    args.iteration = 0
    args.consumed_train_samples = 0

    # Verify indexed dataset order.
    verify_indexed_dataset_order()

    # Datasets.
    print_rank_0(" > data loader.")
    train_data_loader, valid_data_loader, test_data_loader \
        = build_train_valid_test_data_loaders(
            train_valid_test_datasets_provider)

    data_loader_map = {
        "train" : train_data_loader,
        "valid" : valid_data_loader,
        "test" : test_data_loader,
    }

    # Info dict.
    workdir = get_base_pretraining_workdir(args)
    dataset_map = {
        key : {
            "nbr_dir" : os.path.join(workdir, key, "nbr"),
            "data" : GPTChunkDataset(loader.dataset, args.retro_gpt_chunk_length),
        }
        for key, loader in data_loader_map.items() if loader
    }

    return dataset_map
