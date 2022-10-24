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

"""Build an index for similarity search.

Tasks:
- Preprocess chunks
- Embed chunks.
- Index : train, add.
- Neighbors : query, verify.
"""

import json
import os
import torch

from megatron import get_args, initialize_megatron, print_rank_0
from megatron.arguments import _print_args
from tools.retro.db import build_db, preprocess_db, embed_db
from tools.retro.index.build import add_to_index, build_index, train_index
from tools.retro.nbr.build import (
    build_pretraining_neighbors,
    embed_pretraining_chunks,
    query_pretraining_neighbors,
)
from tools.retro.utils import get_args_path, Timer

# >>>
from lutil import pax, print_seq
# <<<


def add_retro_args(parser):
    """Retrieval-LM ('Retro') preprocesing arguments."""

    group = parser.add_argument_group(title = "Retro preprocessing.")

    group.add_argument("--retro-gpt-vocab-file", required = True)
    group.add_argument("--retro-gpt-merge-file", required = True)
    group.add_argument("--retro-gpt-tokenizer-type", required = True)
    group.add_argument("--retro-gpt-seq-length", type = int, required = True)
    group.add_argument("--retro-gpt-chunk-length", type = int, required = True)
    group.add_argument("--retro-bert-vocab-file", required = True)
    group.add_argument("--retro-bert-tokenizer-type", required = True)
    # group.add_argument("--retro-precompute-bert-lengths", action="store_true")
    group.add_argument("--retro-bert-max-chunk-length", type = int, required = True)

    group.add_argument("--retro-tasks", required = True)
    group.add_argument("--retro-index-ty", required = True, choices = [
        "faiss-base",
        "faiss-decomp",
        "faiss-par-add",
    ])
    group.add_argument("--retro-nfeats", "-f", type = int, default = 1024)
    group.add_argument("--retro-nclusters", type = int, required = True)
    group.add_argument("--retro-hnsw-m", type = int, required = True)
    group.add_argument("--retro-ivf-dim", type = int, required = True)
    group.add_argument("--retro-pq-m", type = int, required = True)
    group.add_argument("--retro-pq-nbits", type = int, default = 8)
    group.add_argument("--retro-ef-search", type = int, default = 256)
    group.add_argument("--retro-nprobe", type = int, default = 65536)
    # group.add_argument("--retro-profile-stage-stop", default = None)

    group.add_argument("--retro-workdir", required = True)
    group.add_argument("--retro-nchunks-sampled", type = int, required = True)
    group.add_argument("--retro-block-size", type = int, required = True)
    group.add_argument("--retro-nnbrs-query", type = int, required = True)
    group.add_argument("--retro-nnbrs-target", type = int, required=True)

    return parser


def save_args(args):

    if torch.distributed.get_rank() == 0:
        args_path = get_args_path(args.retro_workdir)
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent = 4, default = lambda o : "<skipped>")

    torch.distributed.barrier()


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider = add_retro_args)

    args = get_args()
    args.retro_tasks = args.retro_tasks.split(",")

    _print_args(args)
    os.makedirs(args.retro_workdir, exist_ok = True)
    save_args(args)

    # Select task to run.
    timer = Timer()
    for task in args.retro_tasks:

        timer.push(task)

        # DB (i.e., chunk db).
        if task == "db-build":
            build_db(args, timer)
        elif task == "db-preprocess":
            preprocess_db(args, timer)
        elif task == "db-embed":
            embed_db(args, timer)

        # Index.
        elif task == "index-build":
            build_index(args, timer) # train, add
        elif task == "index-train":
            train_index(args, timer)
        elif task == "index-add":
            add_to_index(args, timer)
        # elif task == "index-remove-add-outputs":
        elif task == "index-remove-train-files":
            remove_train_files(args, timer)
        elif task == "index-remove-add-files":
            remove_add_files(args, timer)

        # Neighbors.
        elif task == "nbr-build":
            build_pretraining_neighbors(args, timer)
        elif task == "nbr-embed":
            embed_pretraining_chunks(args, timer)
        elif task == "nbr-query":
            query_pretraining_neighbors(args, timer)
        elif task == "nbr-plot-acc":
            plot_nbr_acc(args, timer)
        elif task == "nbr-verify-codes":
            verify_codes(args, timer)
        elif task == "nbr-verify-nbrs":
            verify_nbrs(args, timer)

        # Misc tasks.
        elif task == "misc-time-merge-partials":
            from tools.retro.index import FaissParallelAddIndex
            # if torch.distributed.get_rank() == 0:
            FaissParallelAddIndex.time_merge_partials(args, timer)
            torch.distributed.barrier()
        elif task == "time-hnsw":
            from tools.retro.index import FaissParallelAddIndex
            FaissParallelAddIndex.time_hnsw(args, timer)
        elif task == "time-query":
            from tools.retro.index import FaissParallelAddIndex
            FaissParallelAddIndex.time_query(args, timer)
        elif task == "nan-stats":
            get_nan_stats(args, timer)
        elif task == "bert-nan-analysis":
            run_bert_nan_analysis(args, timer)
        else:
            raise Exception("specialize for task '%s'." % task)

        timer.pop()

        torch.distributed.barrier()

    # Print timing.
    from index.utils import get_index_str
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        # print_rank_0("~~~~~~~~ [ ARG OBJ ] ~~~~~~~~")
        # print_rank_0({k:str(v) for k,v in vars(args).items()})
        print_rank_0("~~~~~~~~ [ TIMER OBJ ] ~~~~~~~~")
        print_rank_0(json.dumps(timer.time_map, indent = 4))
        print_rank_0("~~~~~~~~~~~~~~~~")
        # print_rank_0("[ ARG STR ] = %s" % json.dumps(vars(args)), flush = True)
        print_rank_0("[ TIMER STR ] = %s" % json.dumps(timer.time_map))
        print_rank_0("~~~~~~~~~~~~~~~~")
        timer.print()
        print_rank_0("~~~~~~~~~~~~~~~~")
        print_rank_0("L-RESULT : %s, %s ... %s ... [ %s ]." % (
            args.retro_tasks[-1],
            args.retro_index_ty,
            timer.get_child_str(args.retro_tasks[-1]),
            get_index_str(args),
        ))
    torch.distributed.barrier()
