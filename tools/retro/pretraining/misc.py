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

import numpy as np

from megatron import get_retro_args
from tools.retro.utils import get_gpt_tokenizer

from .retro_dataset import get_retro_datasets

# >>>
from lutil import pax
# <<<


def print_pretraining_neighbors():

    retro_args = get_retro_args()
    gpt_tokenizer = get_gpt_tokenizer()

    train_ds, valid_ds, test_ds = get_retro_datasets()

    def print_tokens(key, token_ids):
        tokens = gpt_tokenizer.detokenize(token_ids)
        print("%s : %s" % (key, "\\n".join(tokens.splitlines())))

    for sample_idx in range(0, len(train_ds), len(train_ds) // 50):

        # >>> ... for fun
        # if sample_idx == 0:
        #     continue
        # <<<

        chunk_index = np.random.randint(train_ds.n_chunks_per_seq)

        header = "################# sample %d, chunk %d. #################" % (
            sample_idx,
            chunk_index,
        )
        print("#" * len(header))
        print(header)
        print("#" * len(header))

        # chunk_idxs = list(range(
        #     sample_idx * train_ds.n_chunks_per_seq,
        #     (sample_idx + 1) * train_ds.n_chunks_per_seq,
        # ))

        sample = train_ds[sample_idx]
        seq_token_ids = sample["text"].tolist()

        chunk_length = retro_args.retro_gpt_chunk_length
        # for chunk_index in range(train_ds.n_chunks_per_seq):
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
        #     "train_ds" : train_ds,
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
