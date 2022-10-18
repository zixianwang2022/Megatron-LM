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
from types import SimpleNamespace

from tools.retro.db.utils import get_db_info_map

# >>>
from lutil import pax
# <<<


def say_hi():
    print("hi, cli.")

# def assert_workdir():
#     assert "WORKDIR" in 
# def get_workdir():
def get_args():
    assert "RETRO_WORKDIR" in os.environ, "Please set environment variable RETRO_WORKDIR (e.g., /path/to/retro/data/)."
    args = SimpleNamespace()
    args.retro_workdir = os.environ["RETRO_WORKDIR"]
    # pax({"args": args})
    return args

def load_chunk_db(args):

    chunk_db_path = get_db_info_map(args)

def print_stats():
    # assert_workdir()
    # workdir = get_workdir()
    args = get_args()

    chunk_db = load_chunk_db(args)
    # chunk_db_sampled = load_chunk_db_sampled()
    # pretraining_dataset = load_pretraining_dataset()

    pax({
        "workdir" : workdir,
        "chunk_db" : chunk_db,
    })

# eof
