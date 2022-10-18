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

from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)

from .timer import Timer


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError as e:
        pass
    return path


def make_sub_dir(top_path, sub_name):
    sub_path = os.path.join(top_path, sub_name)
    mkdir(sub_path)
    return sub_path


def print_rank(*args):
    if len(args) == 1:
        return print_rank(None, *args)
    assert len(args) == 2
    r, s = args
    if r is None or not torch.distributed.is_initialized() or r == torch.distributed.get_rank():
        print("[r %s] ... %s" % (torch.distributed.get_rank() if torch.distributed.is_initialized() else "--", s), flush = True)


def get_args_path(workdir):
    return os.path.join(workdir, "args.json")


def get_gpt_tokenizer(args):
    return _GPT2BPETokenizer(
        vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json",
        merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt",
    )


def get_bert_tokenizer(args):
    return _BertWordPieceTokenizer(
        vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt",
        lower_case = True,
    )
