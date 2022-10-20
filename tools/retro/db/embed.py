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

# import h5py
# import json
# import numpy as np
# import os
# import torch

# from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

# from .dataset import GPTChunkDataset
# from .utils import get_dataset_metas_path, get_db_info_map

# >>>
from lutil import pax
# <<<


# def embed_chunks(args, timer):

#     # Embedding workdir.
#     workdir = os.path.join(args.retro_workdir, "embed")
#     os.makedirs(workdir, exist_ok = True)

#     # Load model.
#     models, optimizer, opt_param_scheduler = \
#         setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

#     # Dataset infos (indexed datasets, chunk index, etc.).
#     dataset_map = get_dataset_map(args)

#     # >>>
#     # del dataset_map["full"]
#     # <<<

#     # Embed each (i.e., full, sampled) dataset.
#     for prefix, dataset in dataset_map.items():
#         print_rank_0(" > embed '%s' chunks. [ count %d ]" %
#                      (prefix, len(dataset)))
#         embed_dataset_chunks(args, workdir, models, prefix, dataset)
# def embed_corpus_chunks(args, timer):
# def embed_chunk_db(args, timer):
def embed_db(args, timer):

    raise Exception("call embed_text_datasets().")

    # Dataset infos (indexed datasets, chunk index, etc.).
    dataset_map = get_dataset_map(args)

    embed_text_datasets(texttttttttttttt)

