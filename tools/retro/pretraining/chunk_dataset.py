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

from megatron import get_args, print_rank_0
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.training import (
    build_train_valid_test_data_loaders,
    update_train_iters,
)
from tools.retro.utils import get_num_chunks_per_seq

from .utils import get_base_pretraining_workdir

# >>>
from lutil import pax
# <<<


class GPTChunkDataset(torch.utils.data.Dataset):

    # def __init__(self, args, seq_dataset):

    #     super().__init__()

    #     self.seq_dataset = seq_dataset

    #     self.seq_length = args.retro_gpt_seq_length
    #     self.chunk_length = args.retro_gpt_chunk_length
    #     assert self.seq_length % self.chunk_length == 0
    #     self.n_chunk_seq_ratio = int(self.seq_length / self.chunk_length)
    #     # self.n_chunks_per_seq = int(self.seq_length / self.chunk_length)

    #     self.n_seqs = len(seq_dataset)
    #     self.n_chunks = self.n_seqs * self.n_chunk_seq_ratio
    def __init__(self, args, seq_dataset):

        super().__init__()

        self.seq_dataset = seq_dataset

        self.n_chunks_per_seq = get_num_chunks_per_seq()
        self.n_seqs = len(seq_dataset)
        self.n_chunks = self.n_seqs * self.n_chunks_per_seq


    def __len__(self):
        return self.n_chunks


    def __getitem__(self, idx):

        seq_idx = idx // self.n_chunk_seq_ratio
        chunk_idx = idx % self.n_chunk_seq_ratio

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


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.retro_gpt_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating pretrained GPT datasets ...")

    return train_ds, valid_ds, test_ds


# def get_text_chunk_dataset_map(args):
def get_gpt_chunk_dataset_map():

    args = get_args()

    # Update train iters.
    update_train_iters(args)

    args.iteration = 0
    args.consumed_train_samples = 0

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
    # text_dataset_map = {
    dataset_map = {
        key : {
            "embed_dir" : os.path.join(workdir, key, "embed"),
            "nbr_dir" : os.path.join(workdir, key, "nbr"),
            # "data" : GPTToTextDataset(GPTChunkDataset(args, loader.dataset)),
            "data" : GPTChunkDataset(args, loader.dataset),
        }
        for key, loader in data_loader_map.items() if loader
    }

    # return text_dataset_map
    return dataset_map
