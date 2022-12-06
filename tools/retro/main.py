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
from megatron.global_vars import set_retro_args
from tools.retro.db import build_db
from tools.retro.db.misc import print_db_embeddings # print_db_neighbors
from tools.retro.index.build import add_to_index, build_index, train_index
from tools.retro.index.misc.megatron_vs_huggingface import run_bert_comparison
from tools.retro.pretraining.query import query_pretraining_neighbors
from tools.retro.pretraining.retro_dataset import test_retro_dataset
from tools.retro.pretraining.misc import print_pretraining_neighbors
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
    group.add_argument("--retro-bert-batch-size", type = int, required = True)
    group.add_argument("--retro-bert-max-chunk-length", type=int, required=True)

    group.add_argument("--retro-tasks", required = True)
    group.add_argument("--retro-index-ty", required = True, choices = [
        "faiss-base",
        "faiss-decomp",
        "faiss-par-add",
    ])
    group.add_argument("--retro-nfeats", "-f", type = int, default = 1024)
    # group.add_argument("--retro-nclusters", type = int, required = True)
    # group.add_argument("--retro-hnsw-m", type = int, required = True)
    # group.add_argument("--retro-ivf-dim", type = int, required = True)
    # group.add_argument("--retro-pq-m", type = int, required = True)
    # group.add_argument("--retro-pq-nbits", type = int, default = 8)
    group.add_argument("--retro-index-str", required = True)
    group.add_argument("--retro-ef-search", type = int, default = 256)
    group.add_argument("--retro-nprobe", type = int, default = 65536)
    # group.add_argument("--retro-profile-stage-stop", default = None)

    # group.add_argument("--retro-workdir", required = True)
    group.add_argument("--retro-nchunks-sampled", type = int, required = True)
    group.add_argument("--retro-doc-block-size", type = int, required = True)
    group.add_argument("--retro-block-size", type = int, required = True)
    group.add_argument("--retro-nnbrs-query", type = int, required = True)
    group.add_argument("--retro-nnbrs-target", type = int, required=True)
    group.add_argument("--retro-nnbrs-pretraining", type = int, required=True)
    # group.add_argument("--retro-embedder", choices = ["megatron", "huggingface"],
    #                    default = "megatron")
    # group.add_argument("--retro-dump-huggingface-embeddings", action="store_true")

    return parser


def save_args(args):

    # args.retro_args = None # matches pretraining format

    if torch.distributed.get_rank() == 0:
        args_path = get_args_path(args.retro_workdir)
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent = 4, default = lambda o : "<skipped>")

    # args.retro_args = args

    torch.distributed.barrier()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def check_index_train_valid_split(timer):

    # >>>
    # # Load chunk db dataset.
    # print_rank_0("load chunk db dataset.")
    # chunk_db_dataset = get_db_merged_train_dataset()
    # # pax(0, {"chunk_db_dataset": chunk_db_dataset})

    # # Load index, banned chunk ids, datasets.
    # print_rank_0(" > get index.")
    # # >>>
    # index = get_index(chunk_db_dataset)
    # <<<

    import faiss
    from tools.retro.db.utils import get_indexed_dataset_infos

    indexed_dataset_infos = get_indexed_dataset_infos()

    index_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/added_0667_0000-0666.faissindex"
    index = faiss.read_index(index_path)

    pax(0, {
        "indexed_dataset_infos" : indexed_dataset_infos,
        "indexed_dataset_infos / 0" : indexed_dataset_infos[0],
        "index_path" : index_path,
        "index" : index,
    })
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider = add_retro_args)

    args = get_args()
    args.retro_tasks = args.retro_tasks.split(",")

    os.makedirs(args.retro_workdir, exist_ok = True)
    save_args(args)
    set_retro_args(args)

    # Select task to run.
    timer = Timer()
    for task in args.retro_tasks:

        timer.push(task)

        # DB (i.e., chunk db).
        if task == "db-build":
            build_db(timer) # preprocess, embed.
        elif task == "db-preprocess":
            preprocess_db(timer)
        elif task == "db-embed":
            embed_db(timer)

        # Index.
        elif task == "index-build":
            build_index(timer) # train, add.
        elif task == "index-train":
            train_index(timer)
        elif task == "index-add":
            add_to_index(timer)
        elif task == "index-remove-train-files":
            remove_train_files(timer)
        elif task == "index-remove-add-files":
            remove_add_files(timer)

        # Pretraining.
        elif task == "pretraining-build":
            build_pretraining_neighbors(timer) # embed, query.
        elif task == "pretraining-embed-chunks":
            embed_pretraining_chunks(timer)
        elif task == "pretraining-query-nbrs":
            query_pretraining_neighbors(timer)
        elif task == "pretraining-test-retro-dataset":
            test_retro_dataset(timer)
        elif task == "nbr-plot-acc":
            plot_nbr_acc(timer)
        elif task == "nbr-verify-codes":
            verify_codes(timer)
        elif task == "nbr-verify-nbrs":
            verify_nbrs(timer)

        # Misc tasks.
        elif task == "misc-time-merge-partials":
            from tools.retro.index import FaissParallelAddIndex
            # if torch.distributed.get_rank() == 0:
            FaissParallelAddIndex.time_merge_partials(timer)
            torch.distributed.barrier()
        elif task == "time-hnsw":
            from tools.retro.index import FaissParallelAddIndex
            FaissParallelAddIndex.time_hnsw(timer)
        elif task == "time-query":
            from tools.retro.index import FaissParallelAddIndex
            FaissParallelAddIndex.time_query(timer)
        elif task == "nan-stats":
            get_nan_stats(timer)
        elif task == "bert-nan-analysis":
            run_bert_nan_analysis(timer)
        elif task == "misc-bert-comparison":
            run_bert_comparison(timer)
        elif task == "misc-check-index-train-valid-split":
            check_index_train_valid_split(timer)
        elif task == "misc-db-print-embeddings":
            print_db_embeddings()
        elif task == "misc-db-print-neighbors":
            print_db_neighbors()
        elif task == "misc-pretraining-print-neighbors":
            print_pretraining_neighbors()
        else:
            raise Exception("specialize for task '%s'." % task)

        timer.pop()

        torch.distributed.barrier()

    # Print timing.
    # from index.utils import get_index_str
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
            # get_index_str(),
            args.retro_index_str,
        ))
    torch.distributed.barrier()
