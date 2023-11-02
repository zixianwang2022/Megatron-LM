# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Preprocess data for Retro.

Stages (see argument '--retro-tasks'):
- Build chunk database (DB).
- Build index (train, add).
- Query pretraining neighbors.
"""

import json
import os
import torch
import types

from megatron import get_args, initialize_megatron, print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.global_vars import set_retro_args
from megatron.core.models.retro.data.db import build_db
# from megatron.core.models.retro.data.index import add_to_index, build_index, train_index
# from megatron.core.models.retro.data.query import query_pretraining_neighbors
from megatron.core.models.retro.data.utils import get_config_path
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


def save_config(config):
    '''Save copy of config within retro workdir.'''

    def default_dump(obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        elif isinstance(obj, (
                types.FunctionType,
                types.LambdaType,
                types.MethodType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
        )):
            return f"<{obj.__name__}>"
        else:
            raise Exception("specialize for <%s>." % type(obj).__name__)

    if torch.distributed.get_rank() == 0:
        config_path = get_config_path(config.retro_workdir)
        with open(config_path, "w") as f:
            json.dump(vars(config), f, indent=4, default=default_dump)

    torch.distributed.barrier()


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider=add_retro_args)

    # Retro preprocessing config.
    config = core_transformer_config_from_args(get_args(), config_class=RetroPreprocessingConfig)

    # Save/set retro config.
    os.makedirs(config.retro_workdir, exist_ok=True)
    save_config(config)
    set_retro_args(config)

    # Select task to run.
    for task in config.retro_tasks:

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
