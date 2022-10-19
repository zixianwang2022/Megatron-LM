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
from tools.retro.db.utils import get_db_info_map, \
    get_indexed_dataset_infos as get_db_indexed_dataset_infos
from tools.retro.db.dataset import \
    get_gpt_chunk_dataset_map as get_db_gpt_chunk_dataset_map
from tools.retro.main import add_retro_args
from tools.retro.nbr.dataset import get_dataset_map as get_pt_dataset_map
from tools.retro.utils import get_args_path, get_bert_tokenizer, get_gpt_tokenizer

# >>>
from lutil import pax
# <<<


def shorten_str(s, n):
    return s if len(s) <= n else "%s ... %s" % (s[:n//2], s[-n//2:])


class retro:

    args = None

    ##############################################
    # initialize.
    ##############################################

    @classmethod
    # def load_args(cls, workdir):
    def init_megatron(cls, workdir):

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
    def init(cls, workdir):

        # Load args.
        cls.init_megatron(workdir)

        cls.gpt_tokenizer = get_gpt_tokenizer(cls.args)
        cls.bert_tokenizer = get_bert_tokenizer(cls.args)

        # Load chunk dbs.
        cls.db_indexed_dataset_infos = get_db_indexed_dataset_infos(cls.args)
        cls.db_gpt_chunk_dataset_map = get_db_gpt_chunk_dataset_map(cls.args)

        # Load pretraining datasets.
        cls.pt_dataset_map = get_pt_dataset_map(cls.args)

        # Print usage.
        cls.print_usage()


    ##############################################
    # utils.
    ##############################################

    @classmethod
    def gpt_to_text(cls, token_ids):
        return cls.gpt_tokenizer.detokenize(token_ids)


    @classmethod
    def text_to_bert(cls, text):
        return cls.bert_tokenizer.tokenize(text)


    ##############################################
    # chunk db.
    ##############################################

    @classmethod
    def get_db_n_indexed_datasets(cls):
        return len(cls.db_indexed_dataset_infos)


    @classmethod
    def get_db_indexed_dataset_infos(cls):
        return [(info["ratio"], info["name"])
                for info in cls.db_indexed_dataset_infos]

    @classmethod
    def get_db_n_chunks(cls, data_key):
        return len(cls.db_gpt_chunk_dataset_map[data_key].chunk_index)


    @classmethod
    def get_db_chunk_gpt(cls, data_key, idx):
        return cls.db_gpt_chunk_dataset_map[data_key][idx]["text"].tolist()


    @classmethod
    def get_db_chunk_bert(cls, data_key, idx):
        return cls.text_to_bert(cls.get_db_chunk_text(data_key, idx))


    @classmethod
    def get_db_chunk_text(cls, data_key, idx):
        return cls.gpt_to_text(cls.get_db_chunk_gpt(data_key, idx))


    @classmethod
    def get_db_chunk_and_continuation_text(cls, data_key, idx):
        gpt_chunks = [
            cls.get_db_chunk_gpt(data_key, idx),
            cls.get_db_chunk_gpt(data_key, idx + 1),
        ]
        text_chunks = [ cls.gpt_to_text(g) for g in gpt_chunks ]
        # pax(0, {
        #     "gpt_chunks" : gpt_chunks,
        #     "text_chunks" : text_chunks,
        # })
        return text_chunks

    
    ##############################################
    # pretraining corpus.
    ##############################################

    @classmethod
    def get_pt_n_seqs_n_chunks(cls, data_key):
        assert data_key in cls.pt_dataset_map, \
            "pretraining set '%s' not found (choices: %s)." % (
                data_key, ", ".join(cls.pt_dataset_map.keys()))
        return (
            len(cls.pt_dataset_map[data_key]["data"].seq_dataset),
            len(cls.pt_dataset_map[data_key]["data"]),
        )


    @classmethod
    def get_pt_n_seqs(cls, data_key):
        return cls.get_pt_n_seqs_n_chunks(data_key)[0]


    @classmethod
    def get_pt_n_chunks(cls, data_key):
        return cls.get_pt_n_seqs_n_chunks(data_key)[1]


    @classmethod
    def get_pt_seq_gpt(cls, data_key, idx):
        return cls.pt_dataset_map[data_key]["data"]. \
            seq_dataset[idx]["text"].tolist()


    @classmethod
    # def get_pt_seq_chunk_gpt(cls, data_key, si, ci):
    def get_pt_chunk_gpt(cls, data_key, si, ci):
        chunk_start_idx = cls.args.retro_chunk_length * ci
        chunk_end_idx = cls.args.retro_chunk_length * (ci + 1)
        return cls.get_pt_seq_gpt(data_key, si) \
            [chunk_start_idx:chunk_end_idx]


    @classmethod
    def get_pt_chunks_gpt(cls, data_key, si):
        n_chunks_per_seq = cls.args.retro_seq_length//cls.args.retro_chunk_length
        return [cls.get_pt_chunk_gpt(data_key, si, ci)
                for ci in range(n_chunks_per_seq) ]


    @classmethod
    def get_pt_chunk_nbrs(cls, data_key, si, ci):
        # >>> [ placeholder. ]
        return [943293, 513093, 141571, 148219, 427765, 718675, 20879, 937286]
        # <<<


    @classmethod
    def get_pt_chunk_banned_doc_ids(cls, data_key, si, ci):
        # >>> [ placeholder. ]
        return [218178, 522268, 839392, 124746, 68821, 799736, 322920, 964471]
        # <<<


    ##############################################
    # usage.
    ##############################################

    @classmethod
    def print_usage(cls):

        print()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("examples ... [ *note*: 'db' = chunk db; 'pt' = pretraining. ]")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

        print()
        print("~~~~ indexed datasets ~~~~")
        print("retro.get_db_n_indexed_datasets() : %s" %
              cls.get_db_n_indexed_datasets())
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
        for data_key in ("full", "sampled"):
            print("retro.get_db_n_chunks('%s') : %d." %
                  (data_key, cls.get_db_n_chunks(data_key)))

        print()
        for sq_key in ("seq", "chunk"):
            for data_key in ("train", "valid"): # , "test"):
                print("retro.get_pt_n_%ss('%s') : %d." % (
                    sq_key, data_key,
                    getattr(cls, f"get_pt_n_{sq_key}s")(data_key)))

        print()
        print("~~~~ tokens, text ~~~~")
        print("retro.get_db_chunk_gpt('full', idx) : %s" %
              shorten_str(str(retro.get_db_chunk_gpt("full", 0)), 50))
        print("retro.get_db_chunk_bert('full', idx) : %s" %
              shorten_str(str(retro.get_db_chunk_bert("full", 0)), 50))
        print("retro.get_db_chunk_text('full', idx) : %s" %
              shorten_str(retro.get_db_chunk_text("full", 0).strip(), 50))
        print("retro.get_db_chunk_and_continuation_text('full', idx) :")
        for i, t in enumerate(retro.get_db_chunk_and_continuation_text("full",0)):
            print("  %s'%s'%s" % (
                "[" if i == 0 else " ",
                shorten_str(t.strip().replace("\n", " "), 50),
                "]" if i == 1 else ",",
            ))

        print()
        print("retro.get_pt_seq_gpt('train', si) : %s" %
              shorten_str(str(retro.get_pt_seq_gpt("train", 0)), 50))
        print("retro.get_pt_chunk_gpt('train', si, ci) : %s" %
              shorten_str(str(retro.get_pt_chunk_gpt("train", 0, 7)),50))
        print("retro.get_pt_chunks_gpt('train', si) :")
        chunks_gpt = retro.get_pt_chunks_gpt("train", 0)
        for i, token_ids in enumerate(chunks_gpt[:2]):
            print("  %s%s," % (
                "[" if i == 0 else " ",
                shorten_str(str(token_ids), 50),
                # "]" if cls.args.retro_seq_length // args.retro_chunk_length - 1 else ",",
            ))
        print("   ...")
        print("   %s]" % shorten_str(str(chunks_gpt[-1]), 50))
        # print("retro.get_pt_seq_text")
        # print("retro.get_pt_seq_chunk_text")

        print()
        print("~~~~ neighbors ~~~~")
        print("retro.get_pt_chunk_nbrs('train', si, ci) : %s" %
              shorten_str(str(cls.get_pt_chunk_nbrs("train", 0, 0)), 50))
        print("retro.get_pt_chunk_banned_doc_ids('train', si, ci) : %s" %
              shorten_str(str(cls.get_pt_chunk_banned_doc_ids("train", 0, 0)),50))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")


# eof
