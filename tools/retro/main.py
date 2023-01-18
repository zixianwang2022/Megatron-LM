# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Preprocess data for Retro.

Tasks:
- Build chunk database (DB).
- Build index (train, add).
- Query pretraining neighbors.
"""

import json
import os
import torch

from megatron import get_args, initialize_megatron, print_rank_0
from megatron.global_vars import set_retro_args
from tools.retro.db import build_db
from tools.retro.db.misc import print_db_embeddings
from tools.retro.index.build import add_to_index, build_index, train_index
from tools.retro.index.misc.megatron_vs_huggingface import (
    compare_bert_full_db,
    compare_bert_partial_db,
    compare_bert_neighbor_dists,
)
from tools.retro.index.misc.verify_codes import verify_codes as verify_index_codes
from tools.retro.pretraining.query import query_pretraining_neighbors
from tools.retro.pretraining.misc import print_pretraining_neighbors
from tools.retro.utils import get_args_path


def add_retro_args(parser):
    """Retro preprocesing arguments."""

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
    group.add_argument("--retro-index-ty", required = True,
                       choices = ["faiss-base", "faiss-par-add"])
    group.add_argument("--retro-nfeats", "-f", type = int, default = 1024)
    group.add_argument("--retro-index-str", required = True)
    group.add_argument("--retro-ef-search", type = int, default = 256)
    group.add_argument("--retro-nprobe", type = int, default = 65536)
    group.add_argument("--retro-nchunks-sampled", type = int, required = True)
    group.add_argument("--retro-doc-block-size", type = int, required = True)
    group.add_argument("--retro-block-size", type = int, required = True)
    group.add_argument("--retro-index-train-block-size",
                       type = int, default=3750000)
    group.add_argument("--retro-num-neighbors-query", type = int, required = True)
    group.add_argument("--retro-num-neighbors-target", type = int, required=True)

    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        assert prefix == "retro", \
            "Retro args must be prefixed with '--retro-*', for consistent " \
            "styling. Please fix '%s'." % ", ".join(action.option_strings)

    return parser


def save_args(args):
    '''Save copy of args within retro workdir.'''

    if torch.distributed.get_rank() == 0:
        args_path = get_args_path(args.retro_workdir)
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent = 4, default = lambda o : "<skipped>")

    torch.distributed.barrier()


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider = add_retro_args)

    # Split retro tasks.
    args = get_args()
    args.retro_tasks = args.retro_tasks.split(",")

    # Save/set retro args.
    os.makedirs(args.retro_workdir, exist_ok = True)
    save_args(args)
    set_retro_args(args)

    # Select task to run.
    for task in args.retro_tasks:

        print_rank_0("start '%s'." % task)

        # Run all stages.
        if task == "build":
            build_db()
            torch.distributed.barrier()
            build_index()
            torch.distributed.barrier()
            query_pretraining_neighbors()

        # DB (i.e., chunk db).
        if task == "db-build":
            build_db()

        # Index.
        elif task == "index-train":
            train_index()
        elif task == "index-add":
            add_to_index()
        elif task == "index-build":
            build_index() # train, add.

        # Pretraining.
        elif task == "pretraining-query-neighbors":
            query_pretraining_neighbors()

        # Misc tasks.
        elif task == "misc-db-print-embeddings":
            print_db_embeddings()
        elif task == "misc-db-print-neighbors":
            print_db_neighbors()
        elif task == "misc-db-nan-stats":
            get_nan_stats()
        elif task == "misc-db-bert-nan-analysis":
            run_bert_nan_analysis()
        elif task == "misc-db-longest-bert-chunks":
            print_db_longest_bert_chunks()

        elif task == "misc-index-remove-train-files":
            remove_train_files()
        elif task == "misc-index-remove-add-files":
            remove_add_files()
        elif task == "misc-index-verify-codes":
            verify_index_codes()
        elif task == "misc-index-megatron-huggingface-comparison-full-db":
            compare_bert_full_db()
        elif task == "misc-index-megatron-huggingface-comparison-partial-db":
            compare_bert_partial_db()
        elif task == "misc-index-megatron-huggingface-comparison-neighbor-dists":
            compare_bert_neighbor_dists()
        elif task == "misc-index-check-train-valid-split":
            check_index_train_valid_split()
        elif task == "misc-pretraining-test-retro-dataset":
            test_retro_dataset()
        elif task == "misc-pretraining-neighbor-plot-acc":
            plot_neighbor_acc()
        elif task == "misc-pretraining-neighbor-verify-neighbors":
            verify_neighbors()
        elif task == "misc-pretraining-time-query":
            from tools.retro.index import FaissParallelAddIndex
            FaissParallelAddIndex.time_query()
        elif task == "misc-pretraining-print-neighbors":
            print_pretraining_neighbors()
        else:
            raise Exception("specialize for task '%s'." % task)

        torch.distributed.barrier()

        print_rank_0("end '%s'." % task)
