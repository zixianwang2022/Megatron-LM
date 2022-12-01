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
import hashlib
import numpy as np
import pickle

from megatron import get_retro_args
from tools.retro.utils import get_gpt_tokenizer

from .retro_dataset import get_retro_datasets

# >>>
from lutil import pax
# <<<

def get_new_chunk_hash_map():
    new_chunk_hash_map = {}
    for chunk_id in tqdm(range(len(new_db_chunk_ds)), "new chunk hash map"):
        chunk = new_db_chunk_ds[chunk_id]["text"]
        chunk_hash = hashlib.sha256(pickle.dumps(chunk.tolist())).hexdigest()
        new_chunk_hash_map = 
        if chunk_id % 100000 == 0:
            print("new db chunk %d." % chunk_id)
        pax(0, {
            "chunk_id" : chunk_id,
            "chunk" : chunk,
            "chunk_hash" : chunk_hash,
        })
    return new_chunk_hash_map

def print_pretraining_neighbors():

    retro_args = get_retro_args()
    gpt_tokenizer = get_gpt_tokenizer()

    train_ds, valid_ds, test_ds = get_retro_datasets()
    new_db_chunk_ds = train_ds.db_chunk_dataset

    old_db_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5"
    old_db = h5py.File(old_db_path, "r")
    old_db_doc_ids = old_db["document_id"]
    old_db_chunks = old_db["chunks"]

    new_chunk_hash_map = get_new_chunk_hash_map()

    pax({
        "train_ds" : train_ds,
        "train_ds / len" : len(train_ds),
        "valid_ds / len" : len(valid_ds),
        
        "new_db_chunks" : str(new_db_chunks.shape),
        "old_db_doc_ids" : str(old_db_doc_ids.shape),
        "old_db_chunks" : str(old_db_chunks.shape),
    })

    def print_tokens(key, token_ids):
        tokens = gpt_tokenizer.detokenize(token_ids)
        print("%s : %s" % (key, "\\n".join(tokens.splitlines())))

    for ds in (train_ds, valid_ds):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # for sample_idx in range(0, len(ds), len(ds) // 50):
        for sample_idx in range(0, len(ds), len(ds) // 10):

            # >>> ... for fun
            # if sample_idx == 0:
            #     continue
            # <<<

            chunk_index = np.random.randint(ds.n_chunks_per_seq)

            header = "################# sample %d, chunk %d. #################" % (
                sample_idx,
                chunk_index,
            )
            print("#" * len(header))
            print(header)
            print("#" * len(header))

            # chunk_idxs = list(range(
            #     sample_idx * ds.n_chunks_per_seq,
            #     (sample_idx + 1) * ds.n_chunks_per_seq,
            # ))

            sample = ds[sample_idx]
            seq_token_ids = sample["text"].tolist()

            chunk_length = retro_args.retro_gpt_chunk_length
            # for chunk_index in range(ds.n_chunks_per_seq):
            # >>>>>>>>>>>>>>>
            chunk_token_ids = seq_token_ids \
                [(chunk_index * chunk_length):((chunk_index + 1) * chunk_length)]

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print_tokens("CHUNK", chunk_token_ids)

            for nbr_index, retrieved_token_ids in \
                enumerate(sample["neighbor_tokens"][chunk_index]):

                nbr_token_ids = retrieved_token_ids[:chunk_length]
                cnt_token_ids = retrieved_token_ids[chunk_length:]
                print()
                print_tokens("NBR", nbr_token_ids)
                print_tokens("CNT", cnt_token_ids)

                # pax(0, {
                #     "sample_idx" : sample_idx,
                #     "chunk_index" : chunk_index,
                #     "nbr_index" : nbr_index,
                #     "seq_token_ids" :
                #     "%d / %s" % (len(seq_token_ids), str(seq_token_ids)),
                #     "chunk_token_ids" :
                #     "%d / %s" % (len(chunk_token_ids), str(chunk_token_ids)),
                #     "retrieved_token_ids" :
                #     "%d / %s"%(len(retrieved_token_ids),str(retrieved_token_ids)),
                # })
            # <<<<<<<<<<<<<<<

            # pax(0, {
            #     "ds" : ds,
            #     "sample" : sample,
            #     "sample_idx" : sample_idx,
            #     "chunk_idxs" : "%d / %s" % (len(chunk_idxs), str(chunk_idxs)),
            # })

    pax(0, {
        "train_ds" : train_ds,
        "valid_ds" : valid_ds,
        "test_ds" : test_ds,
        "train_ds / len" : len(train_ds),
        "valid_ds / len" : len(valid_ds),
        "sample" : sample,
    })


def compare_old_neighbors():

    dddd
