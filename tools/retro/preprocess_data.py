# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Preprocess data for Retro.

Stages (see argument '--retro-tasks'):
- Build chunk database (DB).
- Build index (train, add).
- Query pretraining neighbors.
"""

# >>>
import json
import os
import torch

from megatron import get_args, initialize_megatron, print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.models.retro.data.db import build_db
from megatron.core.models.retro.data.index import add_to_index, train_index
from megatron.core.models.retro.data.config import (
    RetroBertEmbedders,
    RetroGPTChunkDatasets,
    RetroPreprocessingConfig,
    RetroTokenizers,
)
from megatron.core.models.retro.data.query.gpt_chunk_dataset import build_gpt_chunk_datasets_from_gpt_datasets
from megatron.core.models.retro.data.query.multi_split_gpt_dataset import (
    MultiSplitGPTDataset,
    MultiSplitGPTDatasetConfig,
)
from megatron.core.models.retro.data.query.query import query_neighbors
from megatron.core.models.retro.data.query.utils import get_query_dir
from megatron.core.models.retro.data.utils import (
    get_config_path,
    get_gpt_data_dir,
)
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
    _GPTSentencePieceTokenizer,
)
from megatron.training import (
    get_train_valid_test_num_samples,
    # update_train_iters,
)
from pretrain_gpt import is_dataset_built_on_rank
from tools.bert_embedding import BertEmbedder, DiskDataParallelBertEmbedder
from tools.retro.config_utils import add_config_args
# <<<


def add_retro_args(parser):
    group = parser.add_argument_group(title="Retro preprocessing")
    add_config_args(group, RetroPreprocessingConfig)
    return parser


def get_bert_embedders(config):
    mem_embedder = BertEmbedder(
        batch_size = config.retro_bert_batch_size,
        max_bert_seq_length = config.retro_bert_max_chunk_length,
        embedder_type = "megatron",
    )
    return RetroBertEmbedders(
        mem = mem_embedder,
        disk = DiskDataParallelBertEmbedder(mem_embedder, config.retro_block_size),
    )


# >>>
# def get_gpt_datasets(config):

#     # Dataset config.
#     # >>>
#     # data_config = core_multi_split_gpt_dataset_config_from_retro_preprocessing_config(
#     #     config=config,
#     #     split=config.retro_gpt_split,
#     #     return_document_ids=True,
#     #     is_dataset_built_on_rank=is_dataset_built_on_rank,
#     #     custom_data_path=None,
#     # )
#     data_dir = get_gpt_data_dir(config.retro_project_dir)
#     blend = list(config.retro_gpt_data_path)
#     for i in range(len(blend) - 1, -1, -2):
#         blend[i] = os.path.join(data_dir, blend[i])
#     data_config = MultiSplitGPTDatasetConfig(
#         is_built_on_rank=is_dataset_built_on_rank,
#         random_seed=config.retro_gpt_seed,
#         sequence_length=config.retro_gpt_seq_length,
#         blend=blend,
#         split=config.retro_gpt_split,
#         split_preprocessing=config.retro_gpt_split,
#         path_to_cache=config.retro_gpt_data_cache_path,
#         return_document_ids=True,
#     )
#     # <<<

#     # >>>
#     # from lutil import pax
#     # pax("config, data_config")
#     # <<<

#     # GPT datasets.
#     print_rank_0(" > multi-split gpt datasets.")
#     train_valid_test_num_samples = get_train_valid_test_num_samples()
#     train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
#         MultiSplitGPTDataset,
#         train_valid_test_num_samples,
#         data_config,
#     ).build()

#     # >>>
#     datasets = RetroGPTDatasets(
#         train=(train_ds, train_valid_test_num_samples[0]),
#         valid=(valid_ds, train_valid_test_num_samples[1]),
#         test=(test_ds, train_valid_test_num_samples[2]),
#     )
#     # <<<

#     # >>>
#     # from lutil import pax
#     # pax("config, data_config, train_valid_test_num_samples, datasets")
#     # <<<

#     return datasets
def get_gpt_chunk_datasets(config):

    # Reset iteration.
    # >>>
    # config.iteration = 0
    # config.consumed_train_samples = 0
    # <<<

    # Dataset config.
    data_dir = get_gpt_data_dir(config.retro_project_dir)
    blend = list(config.retro_gpt_data_path)
    for i in range(len(blend) - 1, -1, -2):
        blend[i] = os.path.join(data_dir, blend[i])
    data_config = MultiSplitGPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=config.retro_gpt_seed,
        sequence_length=config.retro_gpt_seq_length,
        blend=blend,
        split=config.retro_gpt_split,
        split_preprocessing=config.retro_gpt_split,
        path_to_cache=config.retro_gpt_data_cache_path,
        return_document_ids=True,
    )

    # GPT datasets.
    print_rank_0(" > multi-split gpt datasets.")
    train_valid_test_num_samples = get_train_valid_test_num_samples()
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MultiSplitGPTDataset,
        train_valid_test_num_samples,
        data_config,
    ).build()

    gpt_datasets = {
        "train" : (train_ds, train_valid_test_num_samples[0]),
        "valid" : (valid_ds, train_valid_test_num_samples[1]),
        "test"  : (test_ds, train_valid_test_num_samples[2]),
    }

    # Chunk datasets.
    chunk_datasets = build_gpt_chunk_datasets_from_gpt_datasets(
        project_dir=config.retro_project_dir,
        gpt_datasets=gpt_datasets,
        sample_length=config.retro_gpt_seq_length,
        chunk_length=config.retro_gpt_chunk_length,
    )
    chunk_datasets = RetroGPTChunkDatasets(**chunk_datasets)

    # >>>
    # from lutil import pax
    # pax("gpt_datasets, chunk_datasets")
    # <<<

    return chunk_datasets
# <<<


def get_gpt_tokenizer(config):
    '''GPT (BPE) tokenizer.'''
    tokenizer_type = config.retro_gpt_tokenizer_type
    if tokenizer_type == "GPT2BPETokenizer":
        assert config.retro_gpt_vocab_file and config.retro_gpt_merge_file
        return _GPT2BPETokenizer(
            vocab_file=os.path.join(
                config.retro_project_dir,
                config.retro_gpt_vocab_file,
            ),
            merge_file=os.path.join(
                config.retro_project_dir,
                config.retro_gpt_merge_file,
            ),
        )
    elif tokenizer_type == 'GPTSentencePieceTokenizer':
        assert config.retro_gpt_tokenizer_model is not None
        return _GPTSentencePieceTokenizer(os.path.join(
            config.retro_project_dir,
            config.retro_gpt_tokenizer_model,
        ))
    else:
        raise Exception("unrecognized gpt tokenizer, '%s'." % tokenizer_type)


def get_bert_tokenizer(config):
    '''Bert (Wordpiece) tokenizer.'''
    lower_case = {
        "BertWordPieceLowerCase" : True,
        "BertWordPieceCase" : False,
    }[config.retro_bert_tokenizer_type]
    return _BertWordPieceTokenizer(
        vocab_file=os.path.join(
            config.retro_project_dir,
            config.retro_bert_vocab_file,
        ),
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

    # >>>
    # # Update project-dir-relative paths.
    # args.load = os.path.join(args.retro_project_dir, args.load)
    # if args.vocab_file is not None:
    #     args.vocab_file = os.path.join(args.retro_project_dir, args.vocab_file)
    # if args.merge_file is not None:
    #     args.merge_file = os.path.join(args.retro_project_dir, args.merge_file)
    # if args.tokenizer_model is not None:
    #     args.tokenizer_model = os.path.join(args.retro_project_dir, args.tokenizer_model)
    # <<<

    # Retro config.
    config = core_transformer_config_from_args(
        args, config_class=RetroPreprocessingConfig)

    # Add tools.
    config.retro_bert_embedders = get_bert_embedders(config)
    config.retro_gpt_chunk_datasets = get_gpt_chunk_datasets(config)
    config.retro_tokenizers = get_tokenizers(config)

    # >>>
    # from lutil import pax
    # pax("config")
    # <<<

    return config


def save_config(config):
    '''Save copy of config within retro project dir.'''

    if torch.distributed.get_rank() == 0:

        # GPT config + block size.
        config_subset = {
            k:v for k,v in vars(config).items()
            if k.startswith("retro_gpt") and k != "retro_gpt_chunk_datasets"
        }
        config_subset["retro_block_size"] = config.retro_block_size

        # >>>
        # Neighbor directories.
        query_dir = get_query_dir(config.retro_project_dir)
        config_subset["retro_neighbor_dirs"] = {
            k : (os.path.relpath(v["neighbor_dir"], query_dir) if v is not None else None)
            for k, v in vars(config.retro_gpt_chunk_datasets).items()
        }
        # <<<

        # >>>
        # from lutil import pax
        # pax("config_subset, query_dir", {
        #     "retro_neighbor_dirs" : config_subset["retro_neighbor_dirs"]
        # })
        # <<<

        # Save.
        config_path = get_config_path(config.retro_project_dir)
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

        print_rank_0("start '%s%s'." % (
            "" if config.retro_task_validate is None else "[validate] ",
            task,
        ))

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

        print_rank_0("end '%s%s'." % (
            "" if config.retro_task_validate is None else "[validate] ",
            task,
        ))
