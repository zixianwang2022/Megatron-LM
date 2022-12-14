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

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding import DiskDataParallelBertEmbedder
from tools.retro.db.utils import (
    get_indexed_dataset_infos,
    get_merged_sampled_dataset,
)
from tools.retro.index.factory import IndexFactory
from tools.retro.utils import GPTToTextDataset

from .utils import (
    get_index_dir,
    get_training_data_dir,
    get_training_data_merged,
)


def get_empty_index_path():
    '''Path of empty index.'''
    args = get_retro_args()
    index = IndexFactory.get_index(args.retro_index_ty)
    empty_index_path = index.get_empty_index_path(get_index_dir())
    return empty_index_path


def embed_db():
    '''Embed DB chunks.

    Store chunks in blocks on disk. These blocks will later be merged into
    a single dataset for training the index.
    '''

    # Embed only if index not already trained.
    empty_index_path = get_empty_index_path()
    if os.path.isfile(empty_index_path):
        return

    args = get_retro_args()

    # Get db dataset.
    gpt_dataset = get_merged_sampled_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    # Embed dataset.
    embedder = DiskDataParallelBertEmbedder(args.retro_bert_batch_size,
                                            args.retro_bert_max_chunk_length,
                                            args.retro_block_size)
    embedder.embed_text_dataset("index", get_training_data_dir(), text_dataset)


def train_on_embeddings():
    '''Train index on embedded DB chunks.'''
    args = get_retro_args()
    workdir = get_index_dir()
    index = IndexFactory.get_index(args.retro_index_ty)
    index.train(get_training_data_merged, workdir)


def remove_embeddings():
    '''Remove embeddings after training.'''
    torch.distributed.barrier()
    empty_index_path = get_empty_index_path()
    assert os.path.isfile(empty_index_path)
    remove_embedding_dir(EMBED_KEY)


# >>>
def test_alloc_performance():

    import gc
    from lutil import pax
    from tools.retro.utils import Timer

    if torch.distributed.get_rank() == 0:
        def time_alloc(n):
            timer = Timer()

            timer.push(f"n {n}")
            data = np.empty((n, 1024), dtype = "f4")
            data.fill(0)
            timer.pop()

            del data
            gc.collect()

        for _pow in range(9):
            n = int(np.power(10, _pow))
            time_alloc(n)
        time_alloc(int(300e6))

    torch.distributed.barrier()
    exit()

# <<<


def train_index():
    '''Train index on DB chunks.'''
    # >>>
    # test_alloc_performance()
    # exit()
    # <<<
    embed_db()
    train_on_embeddings()
    # remove_embeddings() # uncomment, or manually remove 'training_data_tmp/'
