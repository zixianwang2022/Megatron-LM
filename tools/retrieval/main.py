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
import torch

from megatron import get_args, initialize_megatron, print_rank_0
from tools.retrieval.db import build_chunk_db, preprocess_chunk_db, embed_chunk_db
from tools.retrieval.index.build import add_to_index, train_index
from tools.retrieval.nbr.build import build_neighbor_table
from tools.retrieval.utils import Timer

# >>>
from lutil import pax, print_seq
# <<<


def add_retrieval_args(parser):
    """Retrieval-LM preprocesing arguments."""

    group = parser.add_argument_group(title="Retrieval preprocessing.")

    group.add_argument("--tasks", required = True)
    group.add_argument("--index-ty", required = True, choices = [
        "faiss-base",
        "faiss-decomp",
        "faiss-par-add",
    ])
    group.add_argument("--nfeats", "-f", type = int, default = 1024)
    group.add_argument("--ncluster", type = int, required = True)
    group.add_argument("--hnsw-m", type = int, required = True)
    group.add_argument("--ivf-dim", type = int, required = True)
    group.add_argument("--pq-m", type = int, required = True)
    group.add_argument("--pq-nbits", type = int, default = 8)
    group.add_argument("--ef-search", type = int, default = 256)
    group.add_argument("--n-probe", type = int, default = 65536)
    # group.add_argument("--profile-stage-stop", default = None)

    group.add_argument("--retro-workdir", required = True)
    group.add_argument("--retro-seq-length", type = int, required = True)
    group.add_argument("--retro-chunk-length", type = int, required = True)
    group.add_argument("--retro-nchunks-sampled", type = int, required = True)
    group.add_argument("--retro-block-size", type = int, required = True)
    group.add_argument("--retro-nnbrs-query", type = int, required = True)
    group.add_argument("--retro-nnbrs-target", type = int, required=True)
    # group.add_argument('--weight', type=float, default=0.5)
    # group.add_argument('--adaptor', action='store_true', default=False)
    # group.add_argument('--return-doc-ids', action='store_true', default=False)
    # group.add_argument('--return-neighbor-ids', action='store_true', default=False)
    # group.add_argument('--add-offset-doc-ids', action='store_true', default=False)
    # group.add_argument('--offset-dict-path', type=str, default='')
    # group.add_argument('--retro-neighbors-path', type=str, required = True) # default='')
    # group.add_argument('--project-size', type=int, default=256)
    # group.add_argument('--stored_params', type=dict, default=dict())
    # group.add_argument('--eval_ppl', action='store_true', default=False)
    # group.add_argument('--workers', type=int, default=100,
    #                    help='Number of worker processes to launch')
    # group.add_argument('--embed-start-index', type=int, default=0,
    #                    help='iteration start')
    # group.add_argument('--embed-end-index', type=int, default=0,
    #                    help='iteration end')

    return parser


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider = add_retrieval_args)

    args = get_args()
    args.tasks = args.tasks.split(",")

    # Select task to run.
    timer = Timer()
    for task in args.tasks:

        timer.push(task)

        # Main tasks.
        if task == "db-build":
            build_chunk_db(args, timer)
        elif task == "embed-chunks":
            embed_chunks(args, timer)
        elif task == "index-train":
            train_index(args, timer)
        elif task == "index-add":
            add_to_index(args, timer)
        elif task == "index-remove-add-outputs":
            remove_add_outputs(args, timer)
        elif task == "index-build":
            build_index(args, timer) # train, add
        elif task == "nbr-build":
            build_neighbor_table(args, timer)
        elif task == "nbr-plot-acc":
            plot_nbr_acc(args, timer)
        elif task == "nbr-verify-codes":
            verify_codes(args, timer)
        elif task == "nbr-verify-nbrs":
            verify_nbrs(args, timer)

        # Misc tasks.
        elif task == "misc-time-merge-partials":
            from tools.retrieval.index import FaissParallelAddIndex
            # if torch.distributed.get_rank() == 0:
            FaissParallelAddIndex.time_merge_partials(args, timer)
            torch.distributed.barrier()
        elif task == "time-hnsw":
            from tools.retrieval.index import FaissParallelAddIndex
            FaissParallelAddIndex.time_hnsw(args, timer)
        elif task == "time-query":
            from tools.retrieval.index import FaissParallelAddIndex
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
        # print("~~~~~~~~ [ ARG OBJ ] ~~~~~~~~")
        # print(json.dumps(vars(args), indent = 4), flush = True) # non-serial
        print("~~~~~~~~ [ TIMER OBJ ] ~~~~~~~~")
        print(json.dumps(timer.time_map, indent = 4), flush = True)
        print("~~~~~~~~~~~~~~~~")
        # print("[ ARG STR ] = %s" % json.dumps(vars(args)), flush = True)
        print("[ TIMER STR ] = %s" % json.dumps(timer.time_map), flush = True)
        print("~~~~~~~~~~~~~~~~")
        timer.print()
        print("~~~~~~~~~~~~~~~~")
        # print("L-RESULT : %s, %s, %s, %d, %d, '%s' ... %s ... [ %s ]." % (
        # print("L-RESULT : %s, %s, %d, %d ... %s ... [ %s ]." % (
        print("L-RESULT : %s, %s ... %s ... [ %s ]." % (
            args.tasks[-1],
            # args.data_ty,
            args.index_ty,
            # args.ntrain,
            # args.nadd,
            # args.profile_stage_stop,
            timer.get_child_str(args.tasks[-1]),
            # args.index_str,
            get_index_str(args),
        ), flush = True)
    torch.distributed.barrier()
