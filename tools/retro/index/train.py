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


# >>>
# from lutil import pax
# from tools.retro.db.utils import get_indexed_dataset_infos
# from .utils import get_training_data_paths

# def merge_embeddings():

#     raise Exception("too much effort/time to merge; will like be slow anyway."

#     args = get_retro_args()

#     merged_path = os.path.join(get_index_dir(), "training_data.bin")

#     if os.path.exists(merged_path):
#         raise Exception("yay.")
#         return

#     # raise Exception("uh oh.")

#     indexed_dataset_infos = get_indexed_dataset_infos()
#     data_paths = get_training_data_paths()
#     data_path_block_size = 100
#     data_path_start_idxs = list(range(0, len(data_paths), data_path_block_size))

#     # pax(0, {"data_path_start_idxs": data_path_start_idxs})

#     n_samples = sum(info["n_chunks_sampled"] for info in indexed_dataset_infos)
#     fp = np.memmap(merged_path, dtype = "f4", mode = "w+",
#                    shape = (n_samples, args.retro_nfeats))

#     # >>>
#     # start_idx = 0
#     # for data_path in tqdm(data_paths, "merge training data"):
#     #     with h5py.File(data_path, "r") as hf:
#     #         fp[start_idx:(start_idx+len(hf["data"]))] = hf["data"]
#     #         start_idx += len(hf["data"])
#     #         fp.flush()
#     # fp.flush()
#     # +++
#     merge_start_idx = 0
#     for data_path_start_idx in data_path_start_idxs:

#         data_path_end_idx = \
#             min(len(data_paths), data_path_start_idx + data_path_block_size)
#         block_data_paths = data_paths[data_path_start_idx:data_path_end_idx]

#         block_n = 0
#         for p in block_data_paths:
#             with h5py.File(p, "r") as hf:
#                 block_n += hf["data"].shape[0]

#         block_data = np.empty((block_n, args.retro_nfeats), dtype = "f4")
#         block_data.fill(0)

#         block_start_idx = 0
#         for p in tqdm(
#                 block_data_paths,
#                 "merge block %d / %d" % (data_path_start_idx, len(data_paths)),
#         ):
#             with h5py.File(p, "r") as hf:
#                 block_data[block_start_idx:(block_start_idx+hf["data"].shape[0])]\
#                     = hf["data"]
#                 block_start_idx += hf["data"].shape[0]

#         fp[merge_start_idx:(merge_start_idx+block_n)] = block_data
#         fp.flush()
#         merge_start_idx += block_n

#         # if True or data_path_start_idx > 0:
#         #     pax(0, {
#         #         "block_data_paths" : block_data_paths,
#         #         "data_path_start_idx" : data_path_start_idx,
#         #         "data_path_end_idx" : data_path_end_idx,
#         #         "block_n" : block_n,
#         #         "block_data / shape" : str(block_data.shape),
#         #         "block_data / start" : str(block_data.flatten()[:10]),
#         #         "block_data / end" : str(block_data.flatten()[-10:]),
#         #     })
#     # <<<

#     pax(0, {
#         "data_paths" : data_paths,
#         "indexed_dataset_infos" : indexed_dataset_infos,
#         "indexed_dataset_infos / 0" : indexed_dataset_infos[0],
#         "merged_path" : merged_path,
#         "n_samples" : n_samples,
#     })
# <<<


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
def test_monolithic_alloc_performance():

    import gc
    from lutil import pax
    from tools.retro.utils import Timer

    assert torch.distributed.get_rank() == 0

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
    time_alloc(int(150e6))
    time_alloc(int(200e6))
    time_alloc(int(250e6))
    time_alloc(int(300e6))

def test_iterative_alloc_performance():

    import time

    assert torch.distributed.get_rank() == 0

    n_feats = 1024
    n_samples = 300000000
    # block_size = 1000000
    block_size = 3750000 # *
    # block_size = 10000000

    data = np.empty((n_samples, n_feats), dtype = "f4")
    # data.fill(0) # ... allocates 1.2TB for real; *essential* for performance

    start_time = time.time()
    for block_start_idx in range(0, n_samples, block_size):

        block_end_idx = min(n_samples, block_start_idx + block_size)
        block = np.zeros((block_end_idx - block_start_idx, n_feats), dtype = "f4")
        data[block_start_idx:block_end_idx] = block

        elapsed_time = time.time() - start_time
        print("block %d / %d ... %.1f min, %.1f min." % (
            block_start_idx // block_size,
            int(np.ceil(n_samples / block_size)),
            elapsed_time / 60,
            elapsed_time * n_samples / block_end_idx / 60,
        ))
# <<<


def train_index():
    '''Train index on DB chunks.'''
    # >>>
    # if torch.distributed.get_rank() == 0:
    #     # test_monolithic_alloc_performance()
    #     test_iterative_alloc_performance()
    # torch.distributed.barrier()
    # exit()
    # <<<
    embed_db()
    # >>>
    # merge_embeddings()
    # <<<
    train_on_embeddings()
    # remove_embeddings() # uncomment, or manually remove 'training_data_tmp/'
