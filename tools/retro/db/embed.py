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

? ? ?

from megatron import get_args, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from tools.bert_embedding import embed_text_datasets
from tools.retro.utils import GPTToTextDataset

# from .dataset import get_gpt_chunk_dataset_map
from .dataset import GPTChunkDataset
from .utils import get_indexed_dataset_infos, get_individual_db

# >>>
from lutil import pax
# <<<


# def embed_db(timer):

#     args = get_args()

#     # GPT, text datasets.
#     gpt_dataset_map = get_gpt_chunk_dataset_map()
#     text_dataset_map = {key:{
#         "data" : GPTToTextDataset(info["data"]),
#         "embed_dir" : info["embed_dir"],
#     } for key, info in gpt_dataset_map.items()}

#     pax(0, {"gpt_dataset_map": gpt_dataset_map})

#     # >>>
#     # del text_dataset_map["full"] # for embedding much smaller sampled dataset.
#     # <<<

#     # Embed text datasets.
#     embed_text_datasets(text_dataset_map,
#                         args.retro_bert_max_chunk_length,
#                         args.retro_block_size)
# def embed_individual_db(indexed_dataset_info):
def embed_indexed_dataset_chunks(ds_id, ds_info):

    args = get_args()

    # Load chunk db & indexed dataset
    chunk_db = get_individual_db(ds_id, ds_info)
    chunk_db[:, 0] = 0 # re-label dataset_id -> 0 (since one dataset at a time)
    indexed_dataset = make_indexed_dataset(ds_info["prefix"], "mmap", True)

    # Text dataset.
    gpt_dataset = GPTChunkDataset(
        indexed_datasets = [ indexed_dataset ],
        chunk_db = chunk_db,
        max_gpt_chunk_length = args.retro_gpt_chunk_length,
    )
    text_dataset = GPTToTextDataset(gpt_dataset)
    text_dataset_map = {ds_info["name"]: {
        "data" : text_dataset,
        "embed_dir" : ds_info["embed_dir"],
    }}

    # pax(0, {
    #     "gpt_dataset" : gpt_dataset,
    #     "text_dataset" : text_dataset,
    #     "text_dataset_map" : text_dataset_map,
    #     "text_dataset_map / 0" : text_dataset_map[ds_info["name"]],
    # })

    # Embed text dataset.
    embed_text_datasets(text_dataset_map,
                        args.retro_bert_max_chunk_length,
                        args.retro_block_size)


def embed_db(timer):

    raise Exception("no standalone embed.")

    # Load indexed dataset infos.
    indexed_dataset_infos = get_indexed_dataset_infos()
    # pax(0, {
    #     "indexed_dataset_infos" : indexed_dataset_infos,
    #     "indexed_dataset_infos / 0" : indexed_dataset_infos[0],
    # })

    # Indexed datasets.
    for ds_id, ds_info in enumerate(indexed_dataset_infos):
        print_rank_0("indexed dataset %d / %d [ %s ]." % (
            ds_id,
            len(indexed_dataset_infos),
            ds_info["prefix"],
        ))
        embed_indexed_dataset_chunks(ds_id, ds_info)

