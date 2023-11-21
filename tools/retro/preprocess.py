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
from megatron.core.models.retro.data.index import add_to_index, train_index
from megatron.core.models.retro.data.config import (
    RetroBertEmbedders,
    RetroGPTDatasets,
    RetroPreprocessingConfig,
    RetroTokenizers,
)
from megatron.core.models.retro.data.query import (
    query_neighbors,
    train_valid_test_datasets_provider,
)
from megatron.core.models.retro.data.utils import (
    core_gpt_dataset_config_from_retro_preprocessing_config,
    get_config_path,
)
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
    _GPTSentencePieceTokenizer,
)
from megatron.training import (
    build_train_valid_test_datasets,
    get_train_valid_test_num_samples,
    update_train_iters,
)
from pretrain_gpt import is_dataset_built_on_rank
from tools.bert_embedding import BertEmbedder, DiskDataParallelBertEmbedder

from config_utils import add_config_args


def add_retro_args(parser):
    group = parser.add_argument_group(title="Retro preprocessing")
    add_config_args(group, RetroPreprocessingConfig)
    return parser


def get_bert_embedders(config):
    return RetroBertEmbedders(
        disk = DiskDataParallelBertEmbedder(
            batch_size = config.retro_bert_batch_size,
            max_bert_seq_length = config.retro_bert_max_chunk_length,
            block_size = config.retro_block_size,
            embedder_type = "megatron",
        ),
        mem = BertEmbedder(
            batch_size = config.retro_bert_batch_size,
            max_bert_seq_length = config.retro_bert_max_chunk_length,
            embedder_type = "megatron",
        ),
    )


# >>>
# def get_gpt_datasets(config, train_valid_test_num_iters):
def get_gpt_datasets(config):
# <<<

    # >>>
    # # Reset iterations.
    # config.iteration = 0
    # config.consumed_train_samples = 0
    # <<<

    # Dataset config.
    data_config = core_gpt_dataset_config_from_retro_preprocessing_config(
        config=config,
        is_dataset_built_on_rank=is_dataset_built_on_rank,
    )

    # Datasets.
    print_rank_0(" > datasets.")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        lambda n : train_valid_test_datasets_provider(data_config, n))

    num_train_samples, num_valid_samples, num_test_samples = \
        get_train_valid_test_num_samples()

    datasets = RetroGPTDatasets(
        train=(train_ds, num_train_samples),
        valid=(valid_ds, num_valid_samples),
        test=(test_ds, num_test_samples),
    )

    # >>>
    # pax("config, data_config, datasets", {
    #     f"datasets / {k}" : "%s, %d" % (len(d[0]) if d[0] else None, d[1])
    #     for k,d in vars(datasets).items()
    # }, "num_train_samples, num_valid_samples, num_test_samples", {
    #     "train_iters" : get_args().train_iters,
    #     "valid_iters" : get_args().eval_iters,
    #     # "test_iters" : get_args().test_iters,
    # })
    # <<<

    return datasets


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


def get_tokenizers(config):
    return RetroTokenizers(
        gpt = get_gpt_tokenizer(config),
        bert = get_bert_tokenizer(config),
    )


def get_retro_preprocessing_config():

    # Arguments.
    args = get_args()
    # >>>
    # update_train_iters(args)
    # <<<

    # Retro config.
    config = core_transformer_config_from_args(
        args, config_class=RetroPreprocessingConfig)

    # Add tools.
    config.retro_bert_embedders = get_bert_embedders(config)
    config.retro_gpt_datasets = get_gpt_datasets(config)
    config.retro_tokenizers = get_tokenizers(config)

    return config


def save_config(config):
    '''Save copy of config within retro project dir.'''

    if torch.distributed.get_rank() == 0:
        config_path = get_config_path(config.retro_project_dir)
        config_subset = {
            k:v for k,v in vars(config).items()
            if k.startswith("retro_gpt") and k != "retro_gpt_datasets"
        }
        config_subset["retro_block_size"] = config.retro_block_size
        with open(config_path, "w") as f:
            json.dump(config_subset, f, indent=4, sort_keys=True)

    torch.distributed.barrier()


if __name__ == "__main__":

    # Initalize Megatron.
    initialize_megatron(extra_args_provider=add_retro_args)

    # Retro config.
    config = get_retro_preprocessing_config()

    # Save retro config.
    os.makedirs(config.retro_project_dir, exist_ok=True)
    save_config(config)

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
    for task in tasks:

        print_rank_0("start '%s'." % task)

        # DB (i.e., chunk db).
        if task == "db-build":
            build_db(config)

        # Index.
        elif task == "index-train":
            train_index(config)
        elif task == "index-add":
            add_to_index(config)

        # Query.
        elif task == "query-neighbors":
            query_neighbors(config)

        else:
            raise Exception("specialize for task '%s'." % task)

        torch.distributed.barrier()

        print_rank_0("end '%s'." % task)
