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

from megatron import get_retro_args
from tools.bert_embedding.huggingface import HuggingfaceEmbedder
from tools.retro.utils import get_gpt_tokenizer

from ..retro_dataset import get_retro_datasets
from .align import align_db_idxs, align_pt_idxs
from .pt_neighbors import print_pt_neighbors

# >>>
from lutil import pax
# <<<


def test_old_new():

    args = get_retro_args()
    args.retro_nnbrs_pretraining = 10

    gpt_tokenizer = get_gpt_tokenizer()

    embedder = HuggingfaceEmbedder(128, 256)

    # >>>
    # embedder.embed_text("lawrence.")
    # raise Exception("hi.")
    # <<<

    new_pt_retro_train_ds, new_pt_retro_valid_ds, _ = get_retro_datasets()

    old_db_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5"
    old_db = h5py.File(old_db_path, "r")
    old_db_doc_ids = old_db["document_id"]
    old_db_chunks = old_db["chunks"]

    old_pt_seq_train_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.h5py"
    old_pt_seq_valid_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.h5py"
    old_pt_nbr_train_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.neighbors.wiki.hdf5"
    old_pt_nbr_valid_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.feat.h5py_neighbors.wiki.hdf5"

    # pax({"old pt keys": list(h5py.File(old_pt_seq_train_path, "r").keys())})
    old_pt_seqs_train = h5py.File(old_pt_seq_train_path, "r")["tokens"]
    old_pt_seqs_valid = h5py.File(old_pt_seq_valid_path, "r")["tokens"]
    old_pt_nbrs_train = h5py.File(old_pt_nbr_train_path, "r")["neighbors"]
    old_pt_nbrs_valid = h5py.File(old_pt_nbr_valid_path, "r")["neighbors"]

    old_pt_hash_map, new_pt_hash_map, common_pt_hashes = align_pt_idxs(
        old_pt_seqs_train,
        new_pt_retro_train_ds.chunk_dataset.seq_dataset,
    )
    common_pt_hashes = list(common_pt_hashes)
    np.random.shuffle(common_pt_hashes)

    old_db_hash_map, new_db_hash_map, common_db_hashes = align_db_idxs(
        old_db_chunks,
        new_pt_retro_train_ds.db_chunk_dataset,
    )

    # pax(0, {
    #     "old_pt_hash_map" : len(old_pt_hash_map),
    #     "new_pt_hash_map" : len(new_pt_hash_map),
    #     "common_pt_hashes" : len(common_pt_hashes),
    #     "old_db_hash_map" : len(old_db_hash_map),
    #     "new_db_hash_map" : len(new_db_hash_map),
    #     "common_db_hashes" : len(common_db_hashes),
    # })

    chunk_length = args.retro_gpt_chunk_length
    nnbrs = args.retro_nnbrs_pretraining
    n_chunks_per_seq = new_pt_retro_train_ds.chunk_dataset.n_chunks_per_seq

    # print_db_neighbors(
    #     embedder,
    #     old_db_chunks,
    #     new_pt_retro_train_ds.db_chunk_dataset,
    #     old_db_hash_map,
    #     new_db_hash_map,
    #     common_db_hashes,
    # )
    print_pt_neighbors(
        gpt_tokenizer,
        embedder,
        chunk_length,
        nnbrs,
        n_chunks_per_seq,
        old_db_doc_ids,
        old_db_chunks,
        old_pt_seqs_train,
        old_pt_nbrs_train,
        new_pt_retro_train_ds,
        old_pt_hash_map,
        new_pt_hash_map,
        common_pt_hashes,
        old_db_hash_map,
        new_db_hash_map,
        common_db_hashes,
    )
