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

# >>>
from lutil import pax
# <<<


class retro:

    args = None

    @classmethod
    # def load_full_chunk_db(cls): # args):
    def load_chunk_db_map(cls): # args):

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
            
        # pax({
        #     "db_info_map" : db_info_map,
        #     "db_map" : db_map,
        # })

        return db_map

    # @classmethod
    # def init_args(cls, **kwargs):

    #     # assert "RETRO_WORKDIR" in os.environ, "Please set environment variable RETRO_WORKDIR (e.g., /path/to/retro/data/)."
    #     assert "retro_workdir" in kwargs, "Please set kwargs['retro_workdir'] (e.g., /path/to/retro/data/)."

    #     # Default args.
    #     cls.args = SimpleNamespace()
    #     # cls.args.retro_workdir = os.environ["RETRO_WORKDIR"]
    #     cls.args.retro_workdir = None
    #     cls.args.retro_chunk_length = 64

    #     # User args.
    #     for key, value in kwargs.items():
    #         assert hasattr(cls.args, key)
    #         setattr(cls.args, key, value)

    #     # pax({"args": cls.args})
        
    #     # return cls.args
    @classmethod
    def load_args(cls, workdir):

        # initialize_megatron(extra_args_provider = add_retro_args)


        # Load args.
        args_path = os.path.join(workdir, "args.json")
        assert os.path.exists(args_path), "args.json not found in workdir."
        with open(args_path) as f:
            cls.args = SimpleNamespace(**json.load(f))
            cls.args.retro_workdir = workdir # just in case workdir moved
            cls.args.rank = 0 # int(os.getenv('RANK', '0'))
            cls.args.world_size = 1 # int(os.getenv("WORLD_SIZE", '1'))

        # pax(0, {"args": cls.args})

        set_global_variables(cls.args)
        _initialize_distributed()
        _set_random_seed(cls.args.seed, cls.args.data_parallel_random_init)

        # pax(0, {
        #     "rank" : torch.distributed.get_rank(),
        #     "world_size" : torch.distributed.get_world_size(),
        # })


    @classmethod
    def init(cls, workdir):

        # Load args.
        cls.load_args(workdir)

        # Load chunk db.
        # cls.db_chunk_map = cls.load_chunk_db_map()
        # cls.db_gpt_chunk_dataset_map = get_db_gpt_chunk_dataset_map(cls.args)

        # Load pretraining dataset.
        cls.pretraining_dataset_map = get_pretraining_dataset_map(cls.args)

        pax(0, {
            # "db_chunk_map" : cls.db_chunk_map,
            # "db_gpt_chunk_dataset_map" : cls.db_gpt_chunk_dataset_map,
            "pretraining_dataset_map" : cls.pretraining_dataset_map,
        })

    @classmethod
    def print_stats(cls):
        # assert_workdir()
        # workdir = get_workdir()
        # args = get_args()

        full_chunk_db = load_full_chunk_db(args)
        # sampled_chunk_db = load_sampled_chunk_db(args)
        # pretraining_dataset = load_pretraining_dataset(args)

        pax({
            "args" : args,
            "chunk_db" : chunk_db,
        })

# eof
