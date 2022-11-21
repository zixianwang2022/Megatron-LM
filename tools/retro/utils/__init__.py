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
import torch
import types

from megatron import get_retro_args
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)

from .timer import Timer


# def get_gpt_seq_length_and_return_doc_ids():
# def get_preprocessing_vs_pretraining_args():
#     """*note: hacky ... differentiate between preprocessing & pretraining."""

#     raise Exception("preload workdir args.")

#     # Ideally would consolidate these 2 args; but seq_length==512 (for Bert)
#     # during preprocessing.

#     args = get_args()

#     # Preprocessing.
#     if hasattr(args, "retro_gpt_seq_length"):
#         # return args.retro_gpt_seq_length, True
#         return types.SimpleNamespace(
#             gpt_seq_length = args.retro_gpt_seq_length,
#             gpt_chunk_length = args.retro_gpt_chunk_length,
#             return_doc_ids = True,
#         )

#     # Pretraining.
#     else:
#         # return args.seq_length, False
#         return types.SimpleNamespace(
#             gpt_seq_length = args.seq_length,
#             gpt_chunk_length = args.retro_chunk_length,
#             return_doc_ids = False,
#         )


def get_args_path(workdir):
    return os.path.join(workdir, "args.json")


def get_num_chunks_per_seq():
    args = get_retro_args()
    seq_length = args.retro_gpt_seq_length
    chunk_length = args.retro_gpt_chunk_length
    assert seq_length % chunk_length == 0
    return seq_length // chunk_length
# def get_num_chunks_per_seq():
#     args = get_preprocessing_vs_pretraining_args()
#     assert args.gpt_seq_length % args.gpt_chunk_length == 0
#     return args.gpt_seq_length // args.gpt_chunk_length


def get_gpt_tokenizer():
    args = get_retro_args()
    return _GPT2BPETokenizer(
        vocab_file = args.retro_gpt_vocab_file,
        merge_file = args.retro_gpt_merge_file,
    )


def get_bert_tokenizer():
    args = get_retro_args()
    return _BertWordPieceTokenizer(
        vocab_file = args.retro_bert_vocab_file,
        lower_case = True,
    )


class GPTToTextDataset(torch.utils.data.Dataset):

    def __init__(self, gpt_dataset):

        super().__init__()

        self.gpt_dataset = gpt_dataset
        self.gpt_tokenizer = get_gpt_tokenizer()


    def __len__(self):
        return len(self.gpt_dataset)


    def __getitem__(self, idx):
        gpt_token_ids = self.gpt_dataset[idx]["text"].tolist()
        text = self.gpt_tokenizer.detokenize(gpt_token_ids)
        return {"text": text}
