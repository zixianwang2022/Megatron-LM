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
# import shutil
# import torch

from megatron import get_args
from tools.bert_embedding import embed_text_datasets
from tools.retro.db.utils import get_sampled_merged_dataset
from tools.retro.index.factory import IndexFactory
from tools.retro.utils import GPTToTextDataset

from .utils import (
    clear_embedding_dir,
    get_embedding_dir,
    get_embedding_paths,
    get_index_workdir,
)


# >>>
from lutil import pax
# <<<

EMBED_KEY = "sampled"


def get_empty_index_path():
    args = get_args()
    index = IndexFactory.get_index(args.retro_index_ty)
    empty_index_path = index.get_empty_index_path(get_index_workdir())
    return empty_index_path


def embed_db():

    empty_index_path = get_empty_index_path()
    if os.path.isfile(empty_index_path):
        return

    args = get_args()

    gpt_dataset = get_sampled_merged_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    # clear_embedding_dir(EMBED_KEY)
    embed_text_datasets(
        {"index": {
            "data" : text_dataset,
            "embed_dir" : get_embedding_dir(EMBED_KEY),
        }},
        args.retro_bert_max_chunk_length,
        args.retro_block_size,
    )

    # pax(0, {
    #     "gpt_dataset" : gpt_dataset,
    #     "text_dataset" : text_dataset,
    # })


def train_on_embeddings(timer):
    args = get_args()
    workdir = get_index_workdir()
    input_data_paths = get_embedding_paths(EMBED_KEY)
    index = IndexFactory.get_index(args.retro_index_ty)
    index.train(input_data_paths, workdir, timer)


def remove_embeddings():
    empty_index_path = get_empty_index_path()
    assert os.path.isfile(empty_index_path)
    clear_embedding_dir(EMBED_KEY)


# def train_index(timer):
#     args = get_args()
#     workdir = get_index_workdir()
#     input_data_paths = get_db_info_map()["sampled"]["embed_paths"]
#     index = IndexFactory.get_index(args.retro_index_ty)
#     index.train(input_data_paths, workdir, timer)
def train_index(timer):
    embed_db()
    train_on_embeddings(timer)
    remove_embeddings()
