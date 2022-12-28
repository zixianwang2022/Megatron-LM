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


n_samples = {
    # "train": 100, "valid": 100,
    "train": 1000, "valid": 1000,
    # "train": 10000, "valid": 10000,
    # "train": 100000, "valid": 10000,
}
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
    "approx" : { # n 1000
        "name" : "IVF512_HNSW8,Flat",
        "search" : {"efSearch" : 8, "nprobe" : 32},
    },
    # "approx" : { # n 10000
    #     "name" : "IVF512_HNSW8,Flat",
    #     "search" : {"efSearch" : 8, "nprobe" : 32},
    # },
    # "approx" : { # n 100000
    #     "name" : "IVF4096_HNSW8,Flat",
    #     "search" : {"efSearch" : 8, "nprobe" : 256},
    # },
}

def get_root_dir():
    dirname = os.path.join(
        get_index_dir(),
        "compare",
        "t%d-v%d" % (n_samples["train"], n_samples["valid"]),
    )

def get_datasets():

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

    nbrs = defaultdict(dict)
    for model_key in indexes:
        for index_key, index_info in indexes[model_key].items():

            index = index_info["index"]

            search_params = index_info["search"]
            for k, p in search_params.items():
                faiss.ParameterSpace().set_index_parameter(index, k, p)

            _, _nbrs = index.search(embeddings[model_key]["valid"], max_nbrs)
            nbrs[model_key][index_key] = _nbrs

    # pax(nbrs)

    return nbrs


def get_acc(nbrs):
    acc_map = defaultdict(dict)
    for mkey0 in nbrs:
        for ikey0 in nbrs[mkey0]:
            for mkey1 in nbrs:
                for ikey1 in nbrs[mkey1]:
                    if mkey0 == mkey1 and ikey0 == ikey1 or \
                       mkey0 < mkey1 or ikey0 < ikey1 or \
                       mkey0 != mkey1 and ikey0 != ikey1:
                        continue
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
    from tools.retro.utils import Timer

    args = get_retro_args()
    timer = Timer()

    print("datasets.")
    if timer: timer.push("datasets")
    datasets = get_datasets()
    if timer: timer.pop()

    print("embedders.")
    if timer: timer.push("embedders")
    embedders = get_embedders()
    if timer: timer.pop()

    print("embeddings.")
    if timer: timer.push("embeddings")
    embeddings = get_embeddings(datasets, embedders)
    if timer: timer.pop()

    print("indexes.")
    if timer: timer.push("indexes")
    indexes = get_indexes(embedders)
    if timer: timer.pop()

    print("build.")
    if timer: timer.push("build")
    build_indexes(embeddings, indexes)
    if timer: timer.pop()

    print("search.")
    if timer: timer.push("search")
    nbrs = get_nbrs(embeddings, indexes)
    if timer: timer.pop()

    print("acc.")
    if timer: timer.push("acc")
    acc_map = get_acc(nbrs)
    if timer: timer.pop()

    pax({
        "n_samples" : n_samples,
        "indexes" : sorted(list(set(
            "%s ... %s" % (info["name"], info["search"])
            for imap in indexes.values()
            for info in imap.values()
        ))),
        "acc_map" : acc_map,
        "time_map" : timer.time_map,
    })

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
