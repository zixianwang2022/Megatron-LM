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

import faiss
import numpy as np
import torch

from megatron import get_retro_args
from tools.bert_embedding import BertEmbedder
from tools.retro.db.utils import get_merged_train_dataset
from tools.retro.utils import GPTToTextDataset

from ..acc import rowwise_intersection

# >>>
from lutil import pax
# <<<


def run_bert_comparison():

    from tools.retro.cli import shorten_str

    args = get_retro_args()

    gpt_dataset = get_merged_train_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    megatron_embedder = BertEmbedder(
        args.retro_bert_batch_size,
        args.retro_bert_max_chunk_length,
        force_megatron = True,
    )
    huggingface_embedder = BertEmbedder(
        args.retro_bert_batch_size,
        args.retro_bert_max_chunk_length,
        force_megatron = False,
    )

    n_samples = 10000 # 1024
    text_dataset = torch.utils.data.Subset(text_dataset, range(n_samples))

    megatron_embeds = \
        megatron_embedder.embed_text_dataset(text_dataset, n_samples)
    huggingface_embeds = \
        huggingface_embedder.embed_text_dataset(text_dataset, n_samples)

    megatron_index = faiss.IndexFlatL2(1024)
    huggingface_index = faiss.IndexFlatL2(1024)
    megatron_index.add(megatron_embeds)
    huggingface_index.add(huggingface_embeds)

    max_nbrs = 200
    _, megatron_nbrs = megatron_index.search(megatron_embeds, max_nbrs)
    _, huggingface_nbrs = huggingface_index.search(huggingface_embeds, max_nbrs)

    acc_map = {}
    for n_nbrs in (1, 2, 5, 10, 20, 50, 100, 200):
        intsec = rowwise_intersection(
            megatron_nbrs[:, :n_nbrs],
            huggingface_nbrs[:, :n_nbrs],
        )
        acc_map[n_nbrs] = np.mean(intsec) / n_nbrs

    print("~~~~ megatron nbrs ~~~~")
    print(megatron_nbrs)
    print("~~~~ huggingface nbrs ~~~~")
    print(huggingface_nbrs)
    pax({"acc_map": acc_map})

    pax({
        "gpt_dataset" : gpt_dataset,
        "text_dataset" : text_dataset,
        "text_dataset / 0" : shorten_str(text_dataset[0]["text"], 125),
        "megatron_embedder" : megatron_embedder,
        "huggingface_embedder" : huggingface_embedder,
        "n_samples" : n_samples,
        "megatron_embeds" : megatron_embeds,
        "huggingface_embeds" : huggingface_embeds,
        "megatron_index" : megatron_index,
        "huggingface_index" : huggingface_index,
        "megatron_nbrs" : megatron_nbrs,
        "huggingface_nbrs" : huggingface_nbrs,
    })
