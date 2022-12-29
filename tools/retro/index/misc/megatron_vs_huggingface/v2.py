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
import glob
import json
import numpy as np
import os
import torch
from tqdm import tqdm

from megatron import get_retro_args
from tools.bert_embedding import BertEmbedder, DiskDataParallelBertEmbedder
from tools.bert_embedding.utils import load_data
from tools.retro.db.utils import get_merged_train_dataset
from tools.retro.index.utils import get_index_dir
from tools.retro.utils import GPTToTextDataset, Timer

from ..acc import rowwise_intersection

# >>>
from lutil import pax
# <<<


n_samples = {
    # "train": 100, "valid": 100,
    # "train": 1000, "valid": 1000,
    # "train": 10000, "valid": 10000,
    # "train": 100000, "valid": 100000,
    # "train": 100000, "valid": 10000,
    "train": 1000000, "valid": 10000,
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

    # "approx" : { # n 1000
    #     "name" : "IVF512_HNSW8,Flat",
    #     "search" : {"efSearch" : 8, "nprobe" : 32},
    # },
    # "approx" : { # n 10000
    #     "name" : "IVF512_HNSW8,Flat",
    #     "search" : {"efSearch" : 8, "nprobe" : 32},
    # },
    # "approx" : { # n 100000
    #     "name" : "IVF4096_HNSW8,Flat",
    #     "search" : {"efSearch" : 8, "nprobe" : 256},
    # },
    "approx" : { # n 100000
        "name" : "IVF8192_HNSW8,Flat",
        "search" : {"efSearch" : 8, "nprobe" : 256},
    },
}
max_nbrs = 200


def get_root_dir():
    dirname = os.path.join(
        get_index_dir(),
        "compare",
        "t%d-v%d" % (n_samples["train"], n_samples["valid"]),
    )
    os.makedirs(dirname, exist_ok = True)
    # pax({"dirname": dirname})
    return dirname

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
    # return {
    #     "megatron" : BertEmbedder(
    #         args.retro_bert_batch_size,
    #         args.retro_bert_max_chunk_length,
    #         force_megatron = True,
    #     ),
    #     "huggingface" : BertEmbedder(
    #         args.retro_bert_batch_size,
    #         args.retro_bert_max_chunk_length,
    #         force_megatron = False,
    #     ),
    # }
    return {
        "megatron" : DiskDataParallelBertEmbedder(
            args.retro_bert_batch_size,
            args.retro_bert_max_chunk_length,
            block_size = args.retro_block_size,
            force_megatron = True,
        ),
        "huggingface" : DiskDataParallelBertEmbedder(
            args.retro_bert_batch_size,
            args.retro_bert_max_chunk_length,
            block_size = args.retro_block_size,
            force_megatron = False,
        ),
    }


# def get_embeddings(datasets, embedders):
def get_embeddings(timer):

    datasets = get_datasets()
    embedders = get_embedders()

    embeddings = defaultdict(dict)
    for model_key, embedder in embedders.items():
        for data_key, dataset in datasets.items():
            embed_dir = os.path.join(get_root_dir(), "embed", model_key, data_key)
            embedders[model_key].embed_text_dataset(data_key, embed_dir,
                                                    datasets[data_key])
            embed_paths = sorted(glob.glob(embed_dir + "/*.hdf5"))
            if len(embed_paths) > 1:
                pax({"embed_paths": embed_paths})
            embeddings[model_key][data_key] = load_data(embed_paths)["data"]

    # pax(embeddings)

    return embeddings


# def get_indexes(embedders):
def get_indexes(timer):

    embeddings = get_embeddings(timer)

    indexes = defaultdict(dict)
    for model_key in embeddings:
        for index_key, index_info in index_infos.items():

            index_path = os.path.join(
                get_root_dir(),
                "index",
                "%s_%s.faiss_index" % (model_key, index_key),
            )
            if not os.path.exists(index_path):

                index = faiss.index_factory(1024, index_info["name"])
                data = embeddings[model_key]["train"]

                print("%s, %s ... train index." % (model_key, index_key))
                index.train(data)

                print("%s, %s ... add to index." % (model_key, index_key))
                index.add(data)

                os.makedirs(os.path.dirname(index_path), exist_ok = True)
                faiss.write_index(index, index_path)

            indexes[model_key][index_key] = {
                **index_info,
                "index" : faiss.read_index(index_path),
            }

    # pax(indexes)

    return indexes


# def get_nbrs(embeddings, indexes):
def get_nbrs(timer):

    nbr_path = os.path.join(get_root_dir(), "nbrs.json")
    if not os.path.exists(nbr_path):

        embeddings = get_embeddings(timer)
        indexes = get_indexes(timer)

        timer.push("nbrs")
        nbrs = defaultdict(dict)
        for model_key in indexes:
            for index_key, index_info in indexes[model_key].items():

                index = index_info["index"]

                search_params = index_info["search"]
                for k, p in search_params.items():
                    faiss.ParameterSpace().set_index_parameter(index, k, p)

                _, _nbrs = index.search(embeddings[model_key]["valid"], max_nbrs)
                nbrs[model_key][index_key] = _nbrs.tolist()
        timer.pop()

        with open(nbr_path, "w") as f:
            json.dump(nbrs, f)

    with open(nbr_path) as f:
        nbrs = json.load(f)
        for m in nbrs:
            for i in nbrs[m]:
                nbrs[m][i] = np.array(nbrs[m][i]).astype("i8")

    # pax(nbrs)

    return nbrs


# def get_acc(nbrs):
def get_acc(timer):

    acc_path = os.path.join(get_root_dir(), "accs.json")
    if not os.path.exists(acc_path):

        nbrs = get_nbrs(timer)

        timer.push("accs")
        accs = defaultdict(dict)
        for mkey0 in nbrs:
            for ikey0 in nbrs[mkey0]:
                for mkey1 in nbrs:
                    for ikey1 in nbrs[mkey1]:
                        if mkey0 == mkey1 and ikey0 == ikey1 or \
                           mkey0 < mkey1 or ikey0 < ikey1 or \
                           mkey0 != mkey1 and ikey0 != ikey1:
                            continue
                        pbar = tqdm((1, 2, 5, 10, 20, 50, 100, 200))
                        for n_nbrs in pbar:
                            pbar.set_description("acc %d" % n_nbrs)
                            if n_nbrs > max_nbrs:
                                continue
                            intsec = rowwise_intersection(
                                nbrs[mkey0][ikey0][:, :n_nbrs],
                                nbrs[mkey1][ikey1][:, :n_nbrs],
                            )
                            accs["%s/%s-%s/%s"%(mkey0,ikey0,mkey1,ikey1)][n_nbrs]\
                                = np.mean(intsec) / n_nbrs
        timer.pop()

        with open(acc_path, "w") as f:
            json.dump(accs, f)

    with open(acc_path) as f:
        accs = json.load(f)

    # pax(accs)

    return accs


def run_bert_comparison():

    # from tools.retro.cli import shorten_str

    # args = get_retro_args()
    timer = Timer()

    indexes = get_indexes(timer)
    acc_map = get_acc(timer)

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
