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
from megatron.core.models.retro.data.db import build_db
# from megatron.core.models.retro.data.index import add_to_index, build_index, train_index
# from megatron.core.models.retro.data.query import query_neighbors
from megatron.core.models.retro.data.preprocess import (
    RetroPreprocessingConfig,
    RetroPreprocessingEnv,
)
from megatron.core.models.retro.data.utils import get_config_path
# from megatron.core.transformer import TransformerConfig
from megatron.global_vars import set_retro_args
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
    _GPTSentencePieceTokenizer,
)
from pretrain_gpt import core_gpt_dataset_config_from_args

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


def get_gpt_tokenizer(config):
    '''GPT (BPE) tokenizer.'''
    tokenizer_type = config.retro_gpt_tokenizer_type
    if tokenizer_type == "GPT2BPETokenizer":
        assert config.retro_gpt_vocab_file and config.retro_gpt_merge_file
        return _GPT2BPETokenizer(
            vocab_file=config.retro_gpt_vocab_file,
            merge_file=config.retro_gpt_merge_file,
        )
    elif tokenizer_type == 'GPTSentencePieceTokenizer':
        assert config.retro_gpt_tokenizer_model is not None
        return _GPTSentencePieceTokenizer(config.retro_gpt_tokenizer_model)
    else:
        raise Exception("unrecognized gpt tokenizer, '%s'." % tokenizer_type)


def get_bert_tokenizer(config):
    '''Bert (Wordpiece) tokenizer.'''
    lower_case = {
        "BertWordPieceLowerCase" : True,
        "BertWordPieceCase" : False,
    }[config.retro_bert_tokenizer_type]
    return _BertWordPieceTokenizer(
        vocab_file=config.retro_bert_vocab_file,
        lower_case=lower_case,
    )


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider=add_retro_args)

    # Retro preprocessing config.
    args = get_args()
    config = core_transformer_config_from_args(
        args, config_class=RetroPreprocessingConfig)
    env = RetroPreprocessingEnv(
        config = config,
        data_config = core_gpt_dataset_config_from_args(args),
        gpt_tokenizer = get_gpt_tokenizer(config),
        bert_tokenizer = get_bert_tokenizer(config),
    )

    # Save/set retro config.
    os.makedirs(config.retro_workdir, exist_ok=True)
    save_config(config)
    set_retro_args(config)

    # Expand tasks.
    task_remap = {
        "build" : [ "db-build", "index-train", "index-add", "query-neighbors" ],
        "index-build" : [ "index-train", "index-add" ],
        "db-build" : [ "db-build" ],
        "index-train" : [ "index-train" ],
        "index-add" : [ "index-add" ],
        "query-neighbors" : [ "query-neighbors" ],
    }
    tasks = []
    for task in config.retro_tasks:
        tasks.extend(task_remap[task])
    config.retro_tasks = tasks

    # Select task to run.
    for task in config.retro_tasks:

        print_rank_0("start '%s'." % task)

        # DB (i.e., chunk db).
        if task == "db-build":
            build_db(env)

        # Index.
        elif task == "index-train":
            train_index(env)
        elif task == "index-add":
            add_to_index(env)

        # Query.
        elif task == "query-neighbors":
            query_neighbors(env)

        else:
            raise Exception("specialize for task '%s'." % task)

        torch.distributed.barrier()

        print_rank_0("end '%s'." % task)
