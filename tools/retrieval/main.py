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
- Embed text.
- Train index.
- Add to index.
- Query index.
- Verify neighbors.
"""

import argparse
from datetime import timedelta
# import faiss
# import json
# import numpy as np
import os
# import shutil
# import socket
import torch

# >>>
from lutil import pax, print_rank, print_seq
# <<<

# from tools.retrieval.data import (
#     clean_data,
#     gen_rand_data,
#     get_all_data_paths,
#     get_train_add_data_paths,
# )
# from tools.retrieval.index.factory import IndexFactory
from tools.retrieval.index.utils import (
    # get_index_dir_path,
    get_index_str,
)
# from tools.retrieval.utils import Timer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # Args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required = True)
    parser.add_argument("--nfeats", "-f", type = int, default = 1024)
    parser.add_argument("--ntrain", "-t", type = int, required = True)
    parser.add_argument("--nadd", "-a", type = int, required = True)
    parser.add_argument("--ncluster", type = int, required = True)
    parser.add_argument("--hnsw-m", type = int, required = True) # hnsw-dim
    parser.add_argument("--ivf-dim", type = int, required = True)
    parser.add_argument("--pq-m", type = int, required = True) # pq-dim
    parser.add_argument("--pq-nbits", type = int, default = 8)
    # parser.add_argument("--data-ty", required = True,
    #                     choices = [ "corpus", "wiki", "rand-1m", "rand-100k" ])
    parser.add_argument("--data-dir", required = True)
    parser.add_argument("--index-dir", required = True)
    parser.add_argument("--index-ty", required = True, choices = [
        "faiss-base",
        "faiss-decomp",
        "faiss-par-add",
    ])
    parser.add_argument("--profile-stage-stop", default = None)
    parser.add_argument("--local_rank", type = int, default = None)
    args = parser.parse_args()

    args.index_str = get_index_str(args)
    args.tasks = args.tasks.split(",")

    # # Input data directory. [ ... MOVE TO ARGS ... ]
    # hostname = socket.gethostname()
    # # hostname = os.environ["HOSTNAME_ORIG"]
    # if hostname.startswith("luna-"):
    #     args.base_dir = "/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval"
    # elif hostname.startswith("rno") or hostname.startswith("dracocpu"):
    #     args.base_dir = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval"
    # elif hostname.startswith("ip-"):
    #     args.base_dir = "/mnt/fsx-outputs-chipdesign/lmcafee/retrieval"
    # else:
    #     raise Exception("specialize for hostname '%s'." % hostname)

    # Torch distributed initialization.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    torch.distributed.init_process_group(
        # backend = "nccl",
        backend = "gloo",
        world_size = args.world_size,
        rank = args.rank,
        # timeout = timedelta(minutes = 10),
        timeout = timedelta(days = 1),
    )

    # Get input data batch paths (for training, adding, querying, verifying).
    # if "train" in args.tasks or "add" in args.tasks or "verify" in args.tasks:
    # if any([k in args.tasks for k in ["train", "add", "verify", "query-acc"]]):
    (
        args.ntrain,
        args.nadd,
        args.train_paths,
        args.add_paths,
    ) = get_train_add_data_paths(args)
    args.index_dir_path = get_index_dir_path(args)
    args.index_empty_path = \
        os.path.join(args.index_dir_path, "empty.faissindex")

    pax(0, {"args": args})

    # Select task to run.
    timer = Timer()
    for task in args.tasks:

        torch.distributed.barrier()

        timer.push(task)

        if task == "clean-data":
            clean_data(args, timer)
        elif task == "split-data":
            split_feat_files(args, timer)
        elif task == "gen-rand-data":
            gen_rand_data(args, timer)
        elif task == "train":
            run_train_pipeline(args, timer)
        elif task == "add":
            run_add_pipeline(args, timer)
        elif task == "remove-add-outputs":
            remove_add_outputs(args, timer)
        elif task == "query":
            raise Exception("hi.")
            run_query_pipeline(args, timer)
        elif task == "query-acc":
            run_query_acc_pipeline(args, timer)
        elif task == "time-merge-partials":
            from retrieval.index.faiss_decomp.cluster.ivfpq import IVFPQIndex
            if torch.distributed.get_rank() == 0:
                IVFPQIndex.time_merge_partials(args, timer)
            torch.distributed.barrier()
        elif task == "verify-codes":
            verify_codes(args, timer)
        elif task == "verify-nbrs":
            verify_nbrs(args, timer)
        else:
            raise Exception("specialize for task '%s'." % task)

        timer.pop()

    torch.distributed.barrier()

    # Print timing.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print("~~~~~~~~ [ ARG OBJ ] ~~~~~~~~")
        print(json.dumps(vars(args), indent = 4), flush = True)
        print("~~~~~~~~ [ TIMER OBJ ] ~~~~~~~~")
        print(json.dumps(timer.time_map, indent = 4), flush = True)
        print("~~~~~~~~~~~~~~~~")
        print("[ ARG STR ] = %s" % json.dumps(vars(args)), flush = True)
        print("[ TIMER STR ] = %s" % json.dumps(timer.time_map), flush = True)
        print("~~~~~~~~~~~~~~~~")
        timer.print()
        print("~~~~~~~~~~~~~~~~")
        print("L-RESULT : %s, %s, %s, %d, %d, '%s' ... %s ... [ %s ]." % (
            args.tasks[-1],
            args.data_ty,
            args.index_ty,
            args.ntrain,
            args.nadd,
            args.profile_stage_stop,
            timer.get_child_str(args.tasks[-1]),
            args.index_str,
        ), flush = True)
    torch.distributed.barrier()
