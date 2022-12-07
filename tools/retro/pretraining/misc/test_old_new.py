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
import torch
import types

from megatron import get_retro_args
from tools.bert_embedding.huggingface import HuggingfaceEmbedder
from tools.retro.utils import get_gpt_tokenizer

from ..retro_dataset import get_retro_datasets as get_new_retro_datasets
from .align import align_db_idxs, align_pt_idxs
from .pt_neighbors import print_pt_neighbors

# >>>
from lutil import pax
# <<<


class OldDBDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        f = h5py.File("/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5")
        self.doc_ids = f["document_id"]
        self.chunks = f["chunks"]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {
            "doc_id" : self.doc_ids[idx].item(),
            "text" : np.copy(self.chunks[idx]),
            # "text" : self.chunks[idx],
        }


class OldRetroDataset(torch.utils.data.Dataset):

    def __init__(self, db_ds, token_path, nbr_path):
        super().__init__()
        args = get_retro_args()
        self.db_ds = db_ds
        # pax({"keys": list(h5py.File(token_path).keys())})
        self.tokens = h5py.File(token_path, "r")["tokens"]
        self.nbrs = h5py.File(nbr_path, "r")["neighbors"]
        self.nnbrs = args.retro_nnbrs_pretraining

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, sample_idx):

        tokens = np.copy(self.tokens[sample_idx])
        nbrs = np.copy(self.nbrs[sample_idx][:, :self.nnbrs])
        # tokens = self.tokens[sample_idx]
        # nbrs = self.nbrs[sample_idx][:, :self.nnbrs]
        nbr_token_ids = []
        for ci in range(len(nbrs)):
            crnt_nbr_token_ids = []
            for ni in range(self.nnbrs):
                crnt_nbr_token_ids.append(self.db_ds[nbrs[ci][ni]]["text"])
            # pax({"crnt_nbr_token_ids": crnt_nbr_token_ids})
            nbr_token_ids.append(crnt_nbr_token_ids)
        nbr_token_ids = np.array(nbr_token_ids)

        # pax(0, {
        #     "tokens" : tokens,
        #     "nbrs" : nbrs,
        #     "nbr_token_ids" : nbr_token_ids,
        # })

        return {
            "text" : tokens,
            "neighbor_tokens" : nbr_token_ids,
        }


def get_old_retro_datasets():

    db_ds = OldDBDataset()

    train_ds = OldRetroDataset(
        db_ds,
        "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.h5py",
        "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.neighbors.wiki.hdf5",
    )

    valid_ds = OldRetroDataset(
        db_ds,
        "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.h5py",
        "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.feat.h5py_neighbors.wiki.hdf5",
    )

    # pax(0, {
    #     "train_ds" : train_ds,
    #     "valid_ds" : valid_ds,
    # })

    return train_ds, valid_ds, None


def test_old_new():

    args = get_retro_args()
    args.retro_nnbrs_pretraining = 10

    old_pt_train_ds, old_pt_valid_ds, _ = get_old_retro_datasets()
    new_pt_train_ds, new_pt_valid_ds, _ = get_new_retro_datasets()

    # pax({
    #     "old chunk" : old_pt_train_ds.db_ds[0],
    #     "new chunk" : new_pt_train_ds.db_chunk_dataset[0],
    #     "old sample" : old_pt_train_ds[0],
    #     "new sample" : new_pt_train_ds[0],
    # })

    pt_train_hashes = align_pt_idxs("train", old_pt_train_ds, new_pt_train_ds)
    pt_valid_hashes = align_pt_idxs("valid", old_pt_valid_ds, new_pt_valid_ds)

    # pax({"pt_train_hashes": pt_train_hashes})

    db_hashes = align_db_idxs(
        old_pt_train_ds.db_ds,
        new_pt_train_ds.db_chunk_dataset,
    )

    # pax({"db_hashes": db_hashes})

    meta = types.SimpleNamespace(
        tokenizer = get_gpt_tokenizer(),
        embedder = HuggingfaceEmbedder(128, 256),
        chunk_length = args.retro_gpt_chunk_length,
        nnbrs = args.retro_nnbrs_pretraining,
        n_chunks_per_seq = new_pt_train_ds.chunk_dataset.n_chunks_per_seq,
    )

    # print_db_neighbors(
    #     embedder,
    #     old_db_chunks,
    #     new_pt_retro_train_ds.db_chunk_dataset,
    #     old_db_hash_map,
    #     new_db_hash_map,
    #     common_db_hashes,
    # )
    # print_pt_neighbors(
    #     gpt_tokenizer,
    #     embedder,
    #     chunk_length,
    #     nnbrs,
    #     n_chunks_per_seq,
    #     old_db_doc_ids,
    #     old_db_chunks,
    #     old_pt_seqs_train,
    #     old_pt_nbrs_train,
    #     new_pt_retro_train_ds,
    #     pt_train_hashes,
    #     # db_hashes,
    # )
    print_pt_neighbors(
        meta,
        old_pt_valid_ds,
        new_pt_valid_ds,
        pt_valid_hashes,
        db_hashes,
    )
