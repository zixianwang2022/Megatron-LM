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
import shutil
import torch

from tools.retrieval.index.factory import IndexFactory

# >>>
from lutil import pax
# <<<


def train_index(args, timer):

    # assert torch.cuda.is_available(), "index requires cuda."

    # Embedding workdir.
    workdir = os.path.join(args.retrieval_workdir, "index")
    os.makedirs(workdir, exist_ok = True)

    # Init index.
    # timer.push("init")
    index = IndexFactory.get_index(args)
    # timer.pop()

    # Train index.
    # timer.push("train")
    # index.train(args.train_paths, args.index_dir_path, timer)
    index.train(workdir, timer)
    # timer.pop()

    pax({"index": index})


def add_to_index(args, timer):

    # Init index.
    timer.push("init")
    index = IndexFactory.get_index(args)
    timer.pop()

    pax(0, {"index": index})

    # Add to index.
    timer.push("add")
    output_index_path = index.add(args.add_paths, args.index_dir_path, timer)
    timer.pop()

    return output_index_path


def remove_add_outputs(args, timer):

    # Single process only.
    if torch.distributed.get_rank() != 0:
        return

    # Get file paths.
    add_paths = [
        os.path.join(args.index_dir_path, r, n)
        for r, ds, fs in os.walk(args.index_dir_path)
        for n in [ *ds, *fs ]
        if n.startswith("add")
    ]

    # Remove files.
    for p in add_paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)
        else:
            raise Exception("specialize for this monster, '%s'." % p)

