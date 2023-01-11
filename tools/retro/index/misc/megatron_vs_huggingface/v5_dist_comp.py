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
from tools.retro.db.utils import (
    get_merged_train_dataset,
    get_merged_valid_dataset,
)
from tools.retro.utils import GPTToTextDataset

# >>>
from lutil import pax
# <<<


def get_datasets():
    gpt_datasets = {
        "train" : get_merged_train_dataset(),
        "valid" : get_merged_valid_dataset(),
    }
    text_datasets = { k: GPTToTextDataset(d) for k, d in gpt_datasets.items() }
    return text_datasets


def get_embedders():
    args = get_retro_args()
    return {
        "megatron" : BertEmbedder(
            args.retro_bert_batch_size,
            args.retro_bert_max_chunk_length,
            embedder_type = "megatron",
        ),
        "huggingface" : BertEmbedder(
            args.retro_bert_batch_size,
            args.retro_bert_max_chunk_length,
            embedder_type = "huggingface",
        ),
    }


def get_indexes():

    args = get_retro_args()

    # Read indexes.
    indexes = {
        "megatron" : faiss.read_index("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/added.faissindex", faiss.IO_FLAG_MMAP),
        # "megatron" : faiss.read_index("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki-mt-cased/index/faiss-par-add/IVF262144_HNSW32,Flat/added.faissindex", faiss.IO_FLAG_MMAP),
        "huggingface" : faiss.read_index("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki-hf/index/faiss-par-add/IVF262144_HNSW32,Flat/added.faissindex", faiss.IO_FLAG_MMAP),
    }

    assert len(set([ index.ntotal for index in indexes.values() ])) == 1
    # pax({
    #     "indexes" : indexes,
    #     "imbalance_factor" : {k:i.invlists.imbalance_factor() for k,i in indexes.items()},
    # })

    # Search parameters.
    for index in indexes.values():
        faiss.ParameterSpace().set_index_parameter(index, "efSearch",
                                                   args.retro_ef_search)
        faiss.ParameterSpace().set_index_parameter(index, "nprobe",
                                                   args.retro_nprobe)

    return indexes


class TextListDataset(torch.utils.data.Dataset):
    '''Dataset that holds single string.'''
    def __init__(self, texts):
        assert isinstance(texts, list)
        for text in texts:
            assert isinstance(text, str)
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return {"text": self.texts[i]}


def run_bert_comparison():

    if torch.distributed.get_rank() != 0:
        return

    datasets = get_datasets()
    embedders = get_embedders()
    indexes = get_indexes()
    # max_nbrs = 5
    # max_nbrs = 40
    max_nbrs = 200

    valid_text_subset = torch.utils.data.Subset(
        datasets["valid"],
        range(0, len(datasets["valid"]), len(datasets["valid"]) // 10),
    )
    query_embeddings = {
        k : e.embed_text_dataset(valid_text_subset)
        for k, e in embedders.items()
    }
    nbrs = {
        k : i.search(query_embeddings[k], max_nbrs)[1]
        for k, i in indexes.items()
    }

    from tools.retro.cli import shorten_str
    self_nbr_dists = defaultdict(list)
    cross_nbr_dists = defaultdict(list)
    for valid_idx in range(len(valid_text_subset)):
        print("valid_idx %d / %d." % (valid_idx, len(valid_text_subset)))
        megatron_nbr_ids = nbrs["megatron"][valid_idx]
        huggingface_nbr_ids = nbrs["huggingface"][valid_idx]
        nbr_texts = {
            "megatron" : TextListDataset([datasets["train"][i]["text"]
                                          for i in megatron_nbr_ids]),
            "huggingface" : TextListDataset([datasets["train"][i]["text"]
                                             for i in huggingface_nbr_ids]),
        }
        # pax({
        #     "query text" : shorten_str(valid_text_subset[valid_idx]["text"], 100),
        #     # "megatron_nbr_ids" : megatron_nbr_ids,
        #     # "huggingface_nbr_ids" : huggingface_nbr_ids,
        #     "megatron_nbr_texts" :
        #     [ shorten_str(t, 100) for t in megatron_nbr_texts ],
        #     "huggingface_nbr_texts" :
        #     [ shorten_str(t, 100) for t in huggingface_nbr_texts ],
        # })
        self_nbr_embeddings = {
            "megatron" :
            embedders["megatron"].embed_text_dataset(nbr_texts["megatron"]),
            "huggingface" :
            embedders["huggingface"].embed_text_dataset(nbr_texts["huggingface"]),
        }
        cross_nbr_embeddings = {
            "megatron" :
            embedders["megatron"].embed_text_dataset(nbr_texts["huggingface"]),
            "huggingface" :
            embedders["huggingface"].embed_text_dataset(nbr_texts["megatron"]),
        }
        for k in self_nbr_embeddings:
            # self_nbr_dists[k].append(np.mean([
            self_nbr_dists[k].append([
                np.linalg.norm(query_embeddings[k][valid_idx] - e)
                for e in self_nbr_embeddings[k]])
        for k in cross_nbr_embeddings:
            # cross_nbr_dists[k].append(np.mean([
            cross_nbr_dists[k].append([
                np.linalg.norm(query_embeddings[k][valid_idx] - e)
                for e in cross_nbr_embeddings[k]])

        # >>>
        # for k in self_nbr_embeddings:
        #     print("~~ %s. ~~" % k)
        #     dists = [np.linalg.norm(query_embeddings[k][valid_idx] - e)
        #              for e in self_nbr_embeddings[k]]
        #     dists.sort()
        #     [ print(d) for d in dists ]

        # exit()
        # pax({
        #     "self_nbr_dists" :
        #     {k:[np.linalg.norm(query_embeddings[k][valid_idx] - e)
        #         for e in self_nbr_embeddings[k]]
        #     for k in self_nbr_embeddings},
        # })
        # <<<

    # pax({
    #     "self_nbr_dists" : {k:np.mean(d) for k,d in self_nbr_dists.items()},
    #     "cross_nbr_dists" : {k:np.mean(d) for k,d in cross_nbr_dists.items()},
    # })
    
    # >>>
    for k, dist_lists in self_nbr_dists.items():
        [ dists.sort() for dists in dist_lists ]
    self_nbr_dists = {k:np.mean(dd, axis = 0) for k, dd in self_nbr_dists.items()}
    top_diffs = {k:{
        "top1" : (dd[1] - dd[0]) / dd[0],
        "top2" : (dd[2] - dd[1]) / dd[1],
        "top5" : (dd[5] - dd[4]) / dd[4],
        "top20" : (dd[20] - dd[19]) / dd[19],
        "topn" : (dd[-1] - dd[-2]) / dd[-2],
    } for k, dd in self_nbr_dists.items()}
    # pax(self_nbr_dists)
    pax(top_diffs)
    # <<<

    pax({
        "datasets" : datasets,
        "embedders" : embedders,
        "indexes" : indexes,
    })
