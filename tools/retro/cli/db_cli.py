# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import json
import os
import torch
import types

from megatron.global_vars import set_global_variables, set_retro_args
from megatron.initialize import (
    initialize_megatron,
    _initialize_distributed,
    _set_random_seed,
    # >>>
    _compile_dependencies,
    # <<<
)
from tools.retro.db.utils import (
    get_indexed_dataset_infos as get_db_indexed_dataset_infos,
    get_merged_train_dataset as get_db_dataset,
)
from tools.retro.utils import get_args_path, get_bert_tokenizer, get_gpt_tokenizer


def shorten_str(s, n):
    s = "\\n".join(s.splitlines())
    return s if len(s) <= n else "%s ... %s" % (s[:n//2], s[-n//2:])


class retro:

    args = None

    ##############################################
    # initialize.
    ##############################################

    @classmethod
    def parse_dtype_str(cls, dtype_str):
        return {
            "torch.float16" : torch.float16,
            "torch.float32" : torch.float32,
            "torch.bfloat16" : torch.bfloat16,
        }[dtype_str]

    @classmethod
    def init_megatron(cls, workdir):
        '''Custom initialization of Megatron.'''

        # Load args.
        args_path = get_args_path(workdir)
        assert os.path.exists(args_path), "args.json not found in workdir."
        with open(args_path) as f:
            cls.args = types.SimpleNamespace(**json.load(f))
            cls.args.retro_workdir = workdir # just in case workdir moved
            cls.args.rank = 0 # override env
            cls.args.world_size = 1 # override env
            cls.args.params_dtype = cls.parse_dtype_str(cls.args.params_dtype)

        set_global_variables(cls.args)
        set_retro_args(cls.args)
        _initialize_distributed()
        _set_random_seed(cls.args.seed, cls.args.data_parallel_random_init)
        # >>>
        _compile_dependencies()
        # <<<

    @classmethod
    def init(cls, workdir):
        '''Initialize Megatron, tokenizers, and datasets.'''

        # Load args.
        cls.init_megatron(workdir)

        cls.tokenizers = types.SimpleNamespace(
            gpt=get_gpt_tokenizer(),
            bert=get_bert_tokenizer(),
        )

        # Load data.
        cls.db_indexed_dataset_infos = get_db_indexed_dataset_infos()
        cls.db_dataset = get_db_dataset()

        # Print usage.
        cls.print_usage()

    ##############################################
    # utils.
    ##############################################

    @classmethod
    def gpt_to_text(cls, token_ids):
        '''GPT tokens to text.'''
        return cls.tokenizers.gpt.detokenize(token_ids)

    @classmethod
    def text_to_bert(cls, text):
        '''Text to Bert tokens.'''
        return cls.tokenizers.bert.tokenize(text)

    ##############################################
    # chunk db.
    ##############################################

    @classmethod
    def get_db_num_indexed_datasets(cls):
        '''Number of indexed datasets within blendable dataset.'''
        return len(cls.db_indexed_dataset_infos)

    @classmethod
    def get_db_indexed_dataset_infos(cls):
        '''Dataset infos, including number of training & sampled sets.'''
        return [(info["ratio"], info["name"])
                for info in cls.db_indexed_dataset_infos]

    @classmethod
    def get_db_dataset(cls):
        return cls.db_dataset

    @classmethod
    def get_db_num_chunks(cls):
        '''Number of DB chunks.'''
        return len(cls.get_db_dataset())

    @classmethod
    def get_db_chunk_gpt(cls, idx):
        '''Get DB chunk as GPT token ids.'''
        return cls.get_db_dataset()[idx]["text"].tolist()

    @classmethod
    def get_db_chunk_bert(cls, idx):
        '''Get DB chunk as Bert token ids.'''
        return cls.text_to_bert(cls.get_db_chunk_text(idx))

    @classmethod
    def get_db_chunk_text(cls, idx):
        '''Get DB chunk as text.'''
        return cls.gpt_to_text(cls.get_db_chunk_gpt(idx))

    @classmethod
    def get_db_chunk_and_continuation_text(cls, idx):
        '''Get DB chunk along with continuation, as text.'''

        # Modulus used here to match original implementation (i.e., last
        # chunks continuation wraps around to first chunk).
        return [
            cls.get_db_chunk_text(idx),
            cls.get_db_chunk_text((idx + 1) % len(cls.get_db_dataset())),
        ]

    ##############################################
    # usage.
    ##############################################

    @classmethod
    def print_usage(cls):
        '''Print usage.'''

        print()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("examples ... [ *note*: 'db' = chunk db; 'pt' = pretraining corpus. ]")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

        print()
        print("~~~~ indexed datasets ~~~~")
        print("retro.get_db_num_indexed_datasets() : %s" %
              cls.get_db_num_indexed_datasets())
        print("retro.get_db_indexed_dataset_infos() :")
        for i, (ratio,prefix) in enumerate(cls.get_db_indexed_dataset_infos()):
            print("  %s(%f, %s)%s" % (
                "[" if i == 0 else " ",
                ratio,
                prefix,
                "]" if i == len(cls.db_indexed_dataset_infos) - 1 else ",",
            ))

        print()
        print("~~~~ counts ~~~~")
        print("retro.get_db_num_chunks : %d." % cls.get_db_num_chunks())

        print()
        print("~~~~ tokens, text ~~~~")
        print("retro.get_db_chunk_gpt(chunk_id) : %s" %
              shorten_str(str(retro.get_db_chunk_gpt(0)), 50))
        print("retro.get_db_chunk_bert(chunk_id) : %s" %
              shorten_str(str(retro.get_db_chunk_bert(0)), 50))
        print("retro.get_db_chunk_text(chunk_id) : %s" %
              shorten_str(retro.get_db_chunk_text(0).strip(), 50))
        print("retro.get_db_chunk_and_continuation_text(chunk_id) :")
        for i, t in enumerate(retro.get_db_chunk_and_continuation_text(0)):
            print("  %s'%s'%s" % (
                "[" if i == 0 else " ",
                shorten_str(t.strip().replace("\n", " "), 50),
                "]" if i == 1 else ",",
            ))

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":

    retro.init("/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/next-llm")

    for i in range(0, len(retro.db_dataset), len(retro.db_dataset) // 10):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("gpt tokens  : %s" % str(retro.get_db_chunk_gpt(i)))
        print("bert tokens : %s" % str(retro.get_db_chunk_bert(i)))
        print("text        : %s" % str(retro.get_db_chunk_text(i)))
