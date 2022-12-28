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

from collections import defaultdict
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


def get_datasets(n_samples):

    gpt_dataset = get_merged_train_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)
    text_datasets = {
        "train" :
        torch.utils.data.Subset(text_dataset, range(n_samples["train"])),
        "valid" :
        torch.utils.data.Subset(text_dataset,
                                range(n_samples["train"],
                                      n_samples["train"] + n_samples["valid"])),
    }

    # pax({"text_datasets": text_datasets})

    return text_datasets


def get_embedders():
    args = get_retro_args()
    return {
        "megatron" : BertEmbedder(
            args.retro_bert_batch_size,
            args.retro_bert_max_chunk_length,
            force_megatron = True,
        ),
        "huggingface" : BertEmbedder(
            args.retro_bert_batch_size,
            args.retro_bert_max_chunk_length,
            force_megatron = False,
        ),
    }


def get_embeddings(datasets, embedders):
    return {
        mkey : {
            dkey : embedder.embed_text_dataset(dataset, len(dataset))
            for dkey, dataset in datasets.items()
        }
        for mkey, embedder in embedders.items()
    }


def get_indexes(embedders):
    index_infos = {
        "exact" : {
            "name" : "Flat", # "FlatL2",
            "search" : {},
        },
        # "approx" : args.retro_index_str,
        # "approx" : {
        #     "name" : "IVF262144_HNSW32,Flat",
        #     "search" : {
        #         "efSearch" : 32,
        #         "nprobe" : 4096,
        #     },
        # },
        "approx" : {
            "name" : "IVF512_HNSW8,Flat",
            "search" : {
                "efSearch" : 8,
                "nprobe" : 32,
            },
        },
    }
    # indexes = {
    #     mkey : {
    #         ikey : faiss.index_factory(1024, iinfo["name"])
    #         for ikey, iinfo in index_infos.items()
    #     }
    #     for mkey in embedders
    # }
    indexes = defaultdict(dict)
    for model_key in embedders:
        for index_key, index_info in index_infos.items():
            indexes[model_key][index_key] = {
                **index_info,
                "index" : faiss.index_factory(1024, index_info["name"]),
            }
    # return index_infos, indexes
    return indexes


def build_indexes(embeddings, indexes):
    for model_key in indexes:
        for index_key, index_info in indexes[model_key].items():
            index = index_info["index"]
            data = embeddings[model_key]["train"]
            index.train(data)
            index.add(data)


def get_nbrs(embeddings, indexes):

    max_nbrs = 200

    # nbrs = {
    #     mkey : {
    #         ikey : index.search(embeddings[mkey]["valid"], max_nbrs)
    #         for ikey, index in indexes[mkey].items()
    #     }
    #     for mkey in indexes
    # }
    nbrs = defaultdict(dict)
    for model_key in indexes:
        for index_key, index_info in indexes[model_key].items():

            index = index_info["index"]

            search_params = index_info["search"]
            # if search_params:
            #     pax({"index": index, "search_params": search_params})
            for k, p in search_params.items():
                faiss.ParameterSpace().set_index_parameter(index, k, p)

            _, _nbrs = index.search(embeddings[model_key]["valid"], max_nbrs)
            nbrs[model_key][index_key] = _nbrs

    # pax(nbrs)

    return nbrs


def get_acc(indexes, nbrs):
    acc_map = defaultdict(dict)
    for mkey0 in indexes:
        for ikey0 in indexes[mkey0]:
            for mkey1 in indexes:
                for ikey1 in indexes[mkey1]:
                    if mkey0 == mkey1 and ikey0 == ikey1 or \
                       mkey0 < mkey1 or ikey0 < ikey1 or \
                       mkey0 != mkey1 and ikey0 != ikey1:
                        continue
                    # if mkey0 == mkey1 or ikey0 == ikey1:
                    #     pass
                    # else:
                    #     continue
                    for n_nbrs in (1, 2, 5, 10, 20, 50, 100, 200):
                        intsec = rowwise_intersection(
                            nbrs[mkey0][ikey0][:, :n_nbrs],
                            nbrs[mkey1][ikey1][:, :n_nbrs],
                        )
                        acc_map["%s/%s-%s/%s"%(mkey0,ikey0,mkey1,ikey1)][n_nbrs] \
                            = np.mean(intsec) / n_nbrs
    # pax(acc_map)
    return acc_map


def run_bert_comparison():

    from tools.retro.cli import shorten_str

    args = get_retro_args()

    n_samples = {
        # "train": 100, "valid": 100,
        # "train": 1000, "valid": 1000,
        "train": 10000, "valid": 10000,
    }

    print("datasets.")
    datasets = get_datasets(n_samples)
    # pax(datasets)

    print("embedders.")
    embedders = get_embedders()
    # pax(embedders)

    print("embeddings.")
    embeddings = get_embeddings(datasets, embedders)
    # pax(embeddings)

    print("indexes.")
    indexes = get_indexes(embedders)
    # pax(indexes)

    print("train, add.")
    build_indexes(embeddings, indexes)
    # >>>
    # indexes = [index for imap in indexes.values() for index in imap.values()]
    # pax({str(i):index for i, index in enumerate(indexes)})
    # <<<

    print("search.")
    nbrs = get_nbrs(embeddings, indexes)
    # pax(nbrs)

    print("acc.")
    acc_map = get_acc(indexes, nbrs)
    pax(acc_map)

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
