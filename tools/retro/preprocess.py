# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Preprocess data for Retro.

Stages (see argument '--retro-tasks'):
- Build chunk database (DB).
- Build index (train, add).
- Query pretraining neighbors.
"""

# from dataclasses import dataclass
# import json
# import os
# import torch

from megatron import get_args, initialize_megatron, print_rank_0
from megatron.arguments import core_transformer_config_from_args
# from megatron.global_vars import set_retro_args
# from megatron.core.models.retro.data.db import build_db
# from megatron.core.models.retro.data.index import add_to_index, build_index, train_index
# from megatron.core.models.retro.data.query import query_pretraining_neighbors
# from megatron.core.models.retro.data.utils import get_args_path
# from megatron.core.transformer import TransformerConfig

from config import RetroPreprocessingConfig
from config_utils import add_config_args

# >>>
from lutil import pax
# <<<


def add_retro_args(parser):

    group = parser.add_argument_group(title="Retro preprocessing")
    add_config_args(group, RetroPreprocessingConfig)

    # >>>
    # parser.print_help()
    # raise Exception("hi.")
    # <<<

    return parser


def save_args(args):
    '''Save copy of args within retro workdir.'''

    def default_dump(obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        else:
            raise Exception("specialize for <%s>." % type(obj).__name__)

    if torch.distributed.get_rank() == 0:
        args_path = get_args_path(args.retro_workdir)
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=4, default=default_dump)

    torch.distributed.barrier()


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider=add_retro_args)

    args = get_args()
    config = core_transformer_config_from_args(args, config_class=RetroPreprocessingConfig)

    pax("args, config")
    config = get_retro_config()
    pax("config")

    # >>>
    import argparse

    parser = argparse.ArgumentParser()
    add_retro_args(parser)

    for action in parser._action_groups[0]._actions:
        if "HelpAction" in type(action).__name__:
            continue
        # print("        %s (%s): %s" % (
        #     ", ".join(o.replace("--", "") for o in action.option_strings),
        #     "str" if action.type is None else action.type.__name__,
        #     action.help,
        # ))
        print("    %s: %s = %s" % (
            ", ".join(o.replace("--", "") for o in action.option_strings),
            "str" if action.type is None else action.type.__name__,
            f"'{action.default}'" if isinstance(action.default, str) else action.default, # "None" if action.default is None else action.default,
        ))
        # pax("action")

    exit()

    pax("parser", {
        "_action_groups" : parser._action_groups,
        "_action_groups / 0" : parser._action_groups[0],
        "0 / _actions" : parser._action_groups[0]._actions,
        # "1 / _actions" : parser._action_groups[1]._actions,
        # "2 / _actions" : parser._action_groups[2]._actions,
    })

    config = get_retro_config()

    pax("args, config")
    # <<<

    # Save/set retro args.
    os.makedirs(args.retro_workdir, exist_ok=True)
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
        elif task == "db-build":
            build_db()

        # Index.
        elif task == "index-build":
            build_index() # calls both train + add.
        elif task == "index-train":
            train_index() # train only
        elif task == "index-add":
            add_to_index() # add only

        # Pretraining.
        elif task == "query-pretraining-neighbors":
            query_pretraining_neighbors()

        else:
            raise Exception("specialize for task '%s'." % task)

        torch.distributed.barrier()

        print_rank_0("end '%s'." % task)
