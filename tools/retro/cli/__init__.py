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
import json
import numpy as np
import os
import torch
from types import SimpleNamespace

from megatron.global_vars import set_global_variables
from megatron.initialize import (
    initialize_megatron,
    _initialize_distributed,
    _set_random_seed,
)
# from megatron import get_args, initialize_megatron, print_rank_0
from tools.retro.db.utils import get_db_info_map
from tools.retro.db.dataset import \
    get_gpt_chunk_dataset_map as get_db_gpt_chunk_dataset_map
from tools.retro.main import add_retro_args
from tools.retro.nbr.dataset import get_dataset_map as get_pretraining_dataset_map
from tools.retro.utils import get_args_path, get_bert_tokenizer, get_gpt_tokenizer

# >>>
from lutil import pax
# <<<


def shorten_str(s, n):
    return s if len(s) <= n else "%s ... %s" % (s[:n//2], s[-n//2:])


class retro:

    args = None

    @classmethod
    def load_chunk_db_map(cls):

        db_info_map = get_db_info_map(cls.args)

        db_map = {}
        for key, db_info in db_info_map.items():

            print("load chunk db '%s'." % key)

            f = h5py.File(db_info["db_path"], "r")
            chunk_index = np.copy(f["chunks_valid"])
            f.close()

            db_map[key] = {
                "chunk_index" : chunk_index,
            }
            
        return db_map


    @classmethod
    def load_args(cls, workdir):

        # initialize_megatron(extra_args_provider = add_retro_args)

        # Load args.
        args_path = get_args_path(workdir)
        assert os.path.exists(args_path), "args.json not found in workdir."
        with open(args_path) as f:
            cls.args = SimpleNamespace(**json.load(f))
            cls.args.retro_workdir = workdir # just in case workdir moved
            cls.args.rank = 0 # int(os.getenv('RANK', '0'))
            cls.args.world_size = 1 # int(os.getenv("WORLD_SIZE", '1'))

        set_global_variables(cls.args)
        _initialize_distributed()
        _set_random_seed(cls.args.seed, cls.args.data_parallel_random_init)


    @classmethod
    def print_usage(cls):

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("examples ...")
        print("~~~~~~~~")
        for data_key in ("full", "sampled"):
            print("retro.get_db_n_chunks('%s') : %d." %
                  (data_key, cls.get_db_n_chunks(data_key)))

        for sq_key in ("seq", "chunk"):
            for data_key in ("train", "valid"): # , "test"):
                print("retro.get_pretraining_n_%ss('%s') : %d." % (
                    sq_key, data_key,
                    getattr(cls, f"get_pretraining_n_{sq_key}s")(data_key)))
        print("~~~~~~~~")
        print("retro.get_db_chunk_tokens_gpt('full', idx) : %s" %
              shorten_str(str(retro.get_db_chunk_tokens_gpt("full", 0)), 50))
        print("retro.get_db_chunk_tokens_bert('full', idx) : %s" %
              shorten_str(str(retro.get_db_chunk_tokens_bert("full", 0)), 50))
        print("retro.get_db_chunk_text('full', idx) : %s" %
              shorten_str(retro.get_db_chunk_text("full", 0).strip(), 50))
        print("~~~~~~~~")
        print("retro.get_pretraining_seq_tokens")
        print("retro.get_pretraining_seq_chunk_tokens")
        print("retro.get_pretraining_seq_text")
        print("retro.get_pretraining_seq_chunk_text")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")


    @classmethod
    def init(cls, workdir):

        # Load args.
        cls.load_args(workdir)

        cls.gpt_tokenizer = get_gpt_tokenizer(cls.args)
        cls.bert_tokenizer = get_bert_tokenizer(cls.args)

        # Load chunk db.
        cls.db_gpt_chunk_dataset_map = get_db_gpt_chunk_dataset_map(cls.args)

        # Load pretraining dataset.
        cls.pretraining_dataset_map = get_pretraining_dataset_map(cls.args)

        # Print usage.
        cls.print_usage()


    @classmethod
    def get_db_n_chunks(cls, data_key):
        return len(cls.db_gpt_chunk_dataset_map[data_key].chunk_index)


    @classmethod
    def get_pretraining_n_seqs_n_chunks(cls, data_key):
        assert data_key in cls.pretraining_dataset_map, \
            "pretraining set '%s' not found (choices: %s)." % (
                data_key, ", ".join(cls.pretraining_dataset_map.keys()))
        return (
            len(cls.pretraining_dataset_map[data_key]["data"].seq_dataset),
            len(cls.pretraining_dataset_map[data_key]["data"]),
        )


    @classmethod
    def get_pretraining_n_seqs(cls, data_key):
        return cls.get_pretraining_n_seqs_n_chunks(data_key)[0]


    @classmethod
    def get_pretraining_n_chunks(cls, data_key):
        return cls.get_pretraining_n_seqs_n_chunks(data_key)[1]


    @classmethod
    def get_db_chunk_tokens_gpt(cls, data_key, idx):
        return cls.db_gpt_chunk_dataset_map[data_key][0]["text"].tolist()


    @classmethod
    def get_db_chunk_text(cls, data_key, idx):
        # return cls.db_gpt_chunk_dataset_map[data_key].gpt_tokenizer.detokenize(
        return cls.gpt_tokenizer.detokenize(
            cls.get_db_chunk_tokens_gpt(data_key, idx))

    
    @classmethod
    def get_db_chunk_tokens_bert(cls, data_key, idx):
        return cls.bert_tokenizer.tokenize(
            cls.gpt_tokenizer.detokenize(
                cls.get_db_chunk_tokens_gpt(data_key, idx)))


# eof
