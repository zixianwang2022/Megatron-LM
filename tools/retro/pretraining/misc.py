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
import itertools
import json
import numpy as np
import os
import pickle
from tqdm import tqdm

from megatron import get_retro_args
from tools.retro.utils import get_gpt_tokenizer

from .retro_dataset import get_retro_datasets
from .utils import get_pretraining_workdir

# >>>
from lutil import pax
# <<<

gpt_tokenizer = None

def print_tokens(key, token_ids):

    global gpt_tokenizer
    if gpt_tokenizer is None:
        gpt_tokenizer = get_gpt_tokenizer()

    tokens = gpt_tokenizer.detokenize(token_ids)
    print("%s : %s" % (key, "\\n".join(tokens.splitlines())))

# def get_new_db_chunk_hash_map(chunk_ds):
# def get_new_db_hash_map(chunk_ds):
def get_chunk_ds_hashes(filename, chunk_ds):

    path = os.path.join(
        get_pretraining_workdir(),
        "compare_nbrs",
        filename + ".json",
    )

    # pax(0, {"path": path})

    if os.path.exists(path):
        raise Exception("exists.")
        with open(path) as f:
            return json.load(f)

    os.makedirs(os.path.dirname(path), exist_ok = True)

    hashes = []
    for chunk_id in tqdm(range(len(chunk_ds)), "chunk hashes / %s" % filename):
        # >>>
        if chunk_id == 1000000:
            # pax(0, {"hashes / len": len(hashes)})
            break
        # <<<
        chunk = chunk_ds[chunk_id]["text"]
        hashes.append(hashlib.sha256(pickle.dumps(chunk.tolist())).hexdigest())

    with open(path, "w") as f:
        json.dump(hashes, f)

    # pax(0, {"hashes[:10]" : hashes[:10]})

    return hashes

def test_old_new():

    args = get_retro_args()
    args.retro_nnbrs_pretraining = 10

    # new_db_chunk_ds, new_pt_chunk_

    retro_args = get_retro_args()
    gpt_tokenizer = get_gpt_tokenizer()

    new_pt_retro_train_ds, new_pt_retro_valid_ds, _ = get_retro_datasets()
    # new_db_chunk_ds = train_ds.db_chunk_dataset
    # new_pt_chunk_ds = train_ds.chunk_dataset

    # new_db_hashes = get_chunk_ds_hashes("new_db", new_db_chunk_ds)
    # new_pt_hashes = get_chunk_ds_hashes("new_pt", new_pt_chunk_ds)
    # raise Exception("hi.")

    old_db_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5"
    old_db = h5py.File(old_db_path, "r")
    old_db_doc_ids = old_db["document_id"]
    old_db_chunks = old_db["chunks"]

    old_pt_seq_train_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.h5py"
    old_pt_seq_valid_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.h5py"
    old_pt_nbr_train_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.neighbors.wiki.hdf5"
    old_pt_nbr_valid_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.feat.h5py_neighbors.wiki.hdf5"

    old_pt_seqs_train = h5py.File(old_pt_seq_train_path, "r")["tokens"]
    old_pt_seqs_valid = h5py.File(old_pt_seq_valid_path, "r")["tokens"]
    old_pt_nbrs_train = h5py.File(old_pt_nbr_train_path, "r")["neighbors"]
    old_pt_nbrs_valid = h5py.File(old_pt_nbr_valid_path, "r")["neighbors"]

    some_list = align_old_new_sample_idxs(old_pt_seqs_train,new_pt_retro_train_ds)
    pax(0, {"some_list": some_list})

    chunk_length = args.retro_gpt_chunk_length
    nnbrs = args.retro_nnbrs_pretraining
    n_chunks_per_seq = new_pt_retro_train_ds.chunk_dataset.n_chunks_per_seq

    for sample_idx in range(10): # range(10, 20):

        old_seq = old_pt_seqs_train[sample_idx]
        new_sample = new_pt_retro_train_ds[sample_idx]
        new_seq = new_sample["text"]

        # pax(0, {"old_pt_nbrs_train": old_pt_nbrs_train})

        old_nbr_ids = old_pt_nbrs_train[sample_idx][:, :nnbrs]
        new_nbrs = new_sample["neighbor_tokens"]
        assert nnbrs == new_nbrs.shape[1]

        chunk_idx = np.random.randint(n_chunks_per_seq)
        # for chunk_idx in range(n_chunks_per_seq):

        header = "############## sample %d, chunk %d ##############" % (
            sample_idx, chunk_idx)
        print("#" * len(header))
        print(header)
        print("#" * len(header))
        print_tokens("OLD_CHUNK", old_seq[
            (chunk_idx * chunk_length):((chunk_idx + 1) * chunk_length)])
        print_tokens("NEW_CHUNK", new_seq[
            (chunk_idx * chunk_length):((chunk_idx + 1) * chunk_length)])

        old_nbr_token_ids = []
        new_nbr_token_ids = []
        for nbr_idx in range(nnbrs):

            old_nbr_id = old_nbr_ids[chunk_idx][nbr_idx].item()
            old_nbr_token_ids.append(old_db_chunks[old_nbr_id])
            new_nbr_token_ids.append(new_nbrs[chunk_idx][nbr_idx][:chunk_length])

            # print()
            # print("~~~~~~~~~~~~~~~~~~~~~")
            # print_tokens("OLD_NBR", old_nbr_token_ids)
            # print_tokens("NEW_NBR", new_nbr_token_ids)
        print()
        [ print_tokens("OLD", ts[:20]) for ts in old_nbr_token_ids ]
        print()
        [ print_tokens("NEW", ts[:20]) for ts in new_nbr_token_ids ]

        exit(0)

        # >>>
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # for token_idx, (old_token_id, new_token_id) in \
        #     enumerate(itertools.zip_longest(old_seq, new_seq)):
        #     print("%4d. %5s, %5s ... %10s, %10s." % (
        #         token_idx,
        #         old_token_id,
        #         new_token_id,
        #         gpt_tokenizer.detokenize([ old_token_id ]) if old_token_id else "--",
        #         gpt_tokenizer.detokenize([ new_token_id ]),
        #     ))
        # pax(0, {
        #     "old_pt_seq_train" : "%d / %s" % (len(old_seq), str(old_seq)),
        #     "new_pt_seq_train" : "%d / %s" % (len(new_seq), str(new_seq)),
        #     "old_nbr_chunk_ids" : str(old_nbr_chunk_ids.shape),
        #     # "old_nbrs" : str(old_nbrs.shape),
        #     "new_nbrs" : str(new_nbrs.shape),
        # })
        # <<<

    pax({
        "train_ds" : train_ds,
        "train_ds / len" : len(train_ds),
        "valid_ds / len" : len(valid_ds),
        
        # "new_db_chunks" : str(new_db_chunks.shape),
        # "old_db_doc_ids" : str(old_db_doc_ids.shape),
        # "old_db_chunks" : str(old_db_chunks.shape),

        "old_pt_seqs_train[:10]" : old_pt_seqs_train[:10].tolist(),
        "new_pt_seqs_train / 0" : [ new_pt_chunk_ds.seq_dataset[i]["text"][:2048].tolist() for i in range(10) ],
    })

def print_pretraining_neighbors():

    # >>>
    test_old_new()
    raise Exception("hi.")
    # <<<

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
