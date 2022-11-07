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
import torch
from tqdm import tqdm

from megatron import get_args, print_rank_0
from tools.bert_embedding import DiskDataParallelBertEmbedder
from tools.retro.db.utils import (
    get_indexed_dataset_infos,
    get_sampled_merged_dataset,
)
from tools.retro.index.factory import IndexFactory
from tools.retro.utils import GPTToTextDataset

from .utils import (
    get_index_dir,
    # get_embedding_dir,
    # get_embedding_paths,
    # remove_embedding_dir,
    get_training_data_block_dir,
    get_training_data_block_paths,
    get_training_data_merged_path,
)

# >>>
from lutil import pax
# <<<

# EMBED_KEY = "sampled"


def get_empty_index_path():
    args = get_args()
    index = IndexFactory.get_index(args.retro_index_ty)
    empty_index_path = index.get_empty_index_path(get_index_dir())
    return empty_index_path


def embed_db():

    # Skip embedding if merged data exists.
    merged_data_path = get_training_data_merged_path()
    if os.path.exists(merged_data_path):
        raise Exception("hurrah.")
        return

    # Embed only if index not already trained.
    empty_index_path = get_empty_index_path()
    if os.path.isfile(empty_index_path):
        return

    args = get_args()

    # Get db dataset.
    gpt_dataset = get_sampled_merged_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    # Embed dataset.
    embedder = DiskDataParallelBertEmbedder(args.retro_bert_max_chunk_length,
                                            args.retro_block_size)
    # embedder.embed_text_dataset("index", get_embedding_dir(EMBED_KEY),
    embedder.embed_text_dataset("index", get_training_data_block_dir(),
                                text_dataset)


def merge_embeddings():

    torch.distributed.barrier()
    if torch.distributed.get_rank() != 0:
        return

    args = get_args()

    # merged_path = get_merged_embedding_path(EMBED_KEY)
    merged_path = get_training_data_merged_path()
    if os.path.exists(merged_path):
        raise Exception("hurrah.")
        return

    block_paths = get_training_data_block_paths()
    indexed_dataset_infos = get_indexed_dataset_infos()
    n_merged = sum(info["n_chunks_sampled"] for info in indexed_dataset_infos)

    # with h5py.File(merged_path, "w") as merged_f:
    #     print_rank_0("initialize empty merged data.")
    #     merged_f.create_dataset("data", data = np.zeros(
    #         (n_merged, args.retro_nfeats), dtype = "f4"))
    #     start_idx = 0
    #     for block_idx, block_path in enumerate(block_paths):
    #         print_rank_0("merging block %d / %d." % block_idx, len(block_paths))
    #         with h5py.File(block_path, "r") as block_f:
    #             n_block = len(block_f["data"])
    #             merged_f["data"][start_idx:(start_idx+n_block)] = block_f["data"]
    #             start_idx += n_block
    raise Exception("merge again?")
    with h5py.File(merged_path, "w") as merged_f:

        print_rank_0("initialize empty merged data.")
        merged_data = np.empty((n_merged, args.retro_nfeats), dtype = "f4")

        start_idx = 0
        pbar = tqdm(block_paths)
        for block_idx, block_path in enumerate(pbar):
            # >>>
            # if block_idx == 10:
            #     break
            # <<<
            # if block_idx % 50 == 0:
            #     print_rank_0("merging block %d / %d." %
            #                  (block_idx, len(block_paths)))
            pbar.set_description("merging blocks")
            with h5py.File(block_path, "r") as block_f:
                n_block = len(block_f["data"])
                merged_data[start_idx:(start_idx+n_block)] = block_f["data"]
                start_idx += n_block

        print_rank_0("write merged data.")
        merged_f.create_dataset("data", data = merged_data)

    pax(0, {
        "merged_path" : merged_path,
        "block_paths" : block_paths,
        "indexed_dataset_infos" : indexed_dataset_infos,
        "n_merged" : n_merged,
    })


def train_on_embeddings(timer):
    args = get_args()
    workdir = get_index_dir()
    input_data_paths = get_embedding_paths(EMBED_KEY)
    index = IndexFactory.get_index(args.retro_index_ty)
    index.train(input_data_paths, workdir, timer)


def remove_embeddings():
    torch.distributed.barrier()
    empty_index_path = get_empty_index_path()
    assert os.path.isfile(empty_index_path)
    remove_embedding_dir(EMBED_KEY)


def train_index(timer):
    embed_db()
    merge_embeddings()
    train_on_embeddings(timer)
    # remove_embeddings()
