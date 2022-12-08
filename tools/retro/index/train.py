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
    get_training_data_block_dir,
    get_training_data_block_paths,
    get_training_data_merged,
)

# >>>
from lutil import pax, print_seq
# <<<


def get_empty_index_path():
    args = get_retro_args()
    index = IndexFactory.get_index(args.retro_index_ty)
    empty_index_path = index.get_empty_index_path(get_index_dir())
    return empty_index_path


def embed_db():

    # >>>
    # # Skip embedding if merged data exists.
    # merged_data_path = get_training_data_merged_path()
    # if os.path.exists(merged_data_path):
    #     return
    # <<<

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
    embedder.embed_text_dataset("index", get_training_data_block_dir(),
                                text_dataset)


# def merge_embeddings():

#     torch.distributed.barrier()
#     if torch.distributed.get_rank() != 0:
#         return

#     args = get_retro_args()

#     # Already merged?
#     merged_path = get_training_data_merged_path()
#     if os.path.exists(merged_path):
#         return

#     # Embedding block paths & dataset infos.
#     block_paths = get_training_data_block_paths()
#     indexed_dataset_infos = get_indexed_dataset_infos()
#     n_merged = sum(info["n_chunks_sampled"] for info in indexed_dataset_infos)

#     # Merge embedding blocks.
#     raise Exception("merge again?")
#     with h5py.File(merged_path, "w") as merged_f:

#         # Initialize empty merged data.
#         print_rank_0("initialize empty merged data.")
#         merged_data = np.empty((n_merged, args.retro_nfeats), dtype = "f4")

#         # Read each block.
#         start_idx = 0
#         pbar = tqdm(block_paths)
#         for block_idx, block_path in enumerate(pbar):
#             pbar.set_description("merging blocks")
#             with h5py.File(block_path, "r") as block_f:
#                 n_block = len(block_f["data"])
#                 merged_data[start_idx:(start_idx+n_block)] = block_f["data"]
#                 start_idx += n_block

#         # Write merged data.
#         print_rank_0("write merged data.")
#         merged_f.create_dataset("data", data = merged_data)

#     # pax(0, {
#     #     "merged_path" : merged_path,
#     #     "block_paths" : block_paths,
#     #     "indexed_dataset_infos" : indexed_dataset_infos,
#     #     "n_merged" : n_merged,
#     # })


def train_on_embeddings(timer):
    args = get_retro_args()
    workdir = get_index_dir()
    # input_data_paths = get_embedding_paths(EMBED_KEY)
    # merged_data_path = get_training_data_merged_path()

    # >>>
    # import faiss
    # pax(0, {"threads": faiss.omp_get_max_threads()})
    # print_seq("hi.")
    # <<<

    # input_data = get_training_data_merged()
    # >>>
    # pax(0, {
    #     "input_data / shape" : input_data.shape,
    #     "input_data / head" : input_data[:10],
    #     "input_data / tail" : input_data[-10:],
    # })
    # <<<
    # >>>
    # with h5py.File(merged_data_path, "r") as f:
    #     import time
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     t = time.time()
    #     data = np.copy(f["data"])
    #     t = time.time() - t
    #     print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #     pax(0, {
    #         "shape" : f["data"].shape,
    #         # "head" : f["data"][:10],
    #         # "tail" : f["data"][-10:],
    #         "t" : t,
    #     })
    # <<<
    index = IndexFactory.get_index(args.retro_index_ty)
    # index.train(input_data_paths, workdir, timer)
    # index.train([merged_data_path], workdir, timer)
    # index.train(input_data, workdir, timer)
    index.train(get_training_data_merged, workdir, timer)


def remove_embeddings():
    torch.distributed.barrier()
    empty_index_path = get_empty_index_path()
    assert os.path.isfile(empty_index_path)
    remove_embedding_dir(EMBED_KEY)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def time_training():

#     # index_str = "OPQ64_128,IVF4194304_HNSW32,PQ64"
#     index_str = "OPQ32_256,IVF32768_HNSW32,PQ32"
#     args = get_retro_args()

#     assert torch.distributed.get_rank() == 0

#     # Set num threads (torch.distributed reset it to 1).
#     faiss.omp_set_num_threads(64)

#     empty_index_path = self.get_empty_index_path(dir_path)

#     # Index already exists? -> return.
#     if os.path.isfile(empty_index_path):
#         return

#     # >>>
#     # # Load data.
#     # timer.push("load-data")
#     # inp = load_data(input_data_paths, timer)["data"]
#     # timer.pop()
#     # <<<

#     # print_seq("n_threads = %s." % faiss.omp_get_max_threads())
#     inp = input_data_loader()
#     # pax(0, {"inp": inp})

#     # Init index.
#     timer.push("init")
#     index_str = get_index_str()
#     index = faiss.index_factory(args.retro_nfeats, index_str)
#     timer.pop()

#     # Move to GPU.
#     index_ivf = faiss.extract_index_ivf(index)
#     clustering_index = \
#         faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
#     index_ivf.clustering_index = clustering_index
#     self.c_verbose(index, True)
#     self.c_verbose(index_ivf, True)
#     self.c_verbose(index_ivf.quantizer, True)
#     self.c_verbose(index_ivf.clustering_index, True)

#     # Train index.
#     timer.push("train")
#     index.train(inp)
#     timer.pop()

#     # Save index.
#     timer.push("save")
#     faiss.write_index(index, empty_index_path)
#     timer.pop()
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def train_index(timer):
    # >>>
    # time_training()
    # raise Exception("hi.")
    # <<<
    embed_db()
    # merge_embeddings() # ... deprecated [ non-essential ]
    train_on_embeddings(timer)
    # remove_embeddings()
