# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

? ? ?

"""Utilities for building embedders, datasets, and tokenizers."""

# >>>
# import json
# import os
# import torch
# import types

# from megatron import get_args, initialize_megatron, print_rank_0
# from megatron.arguments import core_transformer_config_from_args
# from megatron.core.models.retro.data.db import build_db
# from megatron.core.models.retro.data.index import add_to_index, train_index
# from megatron.core.models.retro.data.config import (
#     RetroBertEmbedders,
#     RetroGPTDatasets,
#     RetroPreprocessingConfig,
#     RetroTokenizers,
# )
# from megatron.core.models.retro.data.query import (
#     query_neighbors,
#     train_valid_test_datasets_provider,
# )
from megatron.core.models.retro.data.utils import \
    core_gpt_dataset_config_from_retro_preprocessing_config
# from megatron.tokenizer.tokenizer import (
#     _BertWordPieceTokenizer,
#     _GPT2BPETokenizer,
#     _GPTSentencePieceTokenizer,
# )
# from megatron.training import (
#     build_train_valid_test_datasets,
#     get_train_valid_test_num_samples,
#     update_train_iters,
# )
from pretrain_gpt import is_dataset_built_on_rank
# from tools.bert_embedding import BertEmbedder, DiskDataParallelBertEmbedder

# from config_utils import add_config_args
# <<<
