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
# import glob
import json
import numpy as np
import os
import torch
from tqdm import tqdm

from megatron import get_retro_args
from tools.bert_embedding import BertEmbedder
# from tools.bert_embedding.utils import load_data
from tools.retro.db.utils import get_merged_valid_dataset
from tools.retro.index.utils import get_index_dir
from tools.retro.utils import GPTToTextDataset

from ..acc import rowwise_intersection

# >>>
from lutil import pax
# <<<


n_valid = 10
# n_valid = 1000
# n_valid = 10000
max_nbrs = 200
# index_infos = {
#     "exact" : {},
#     "approx" : {
#         "efSearch" : 16,
#         "nprobe" : 4096,
#     },
# }


def get_root_dir():
    dirname = os.path.join(get_index_dir(), "compare")
    os.makedirs(dirname, exist_ok = True)
    return dirname

def get_valid_dataset():
    gpt_dataset = get_merged_valid_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)
    idxs = range(0, len(text_dataset), len(text_dataset) // n_valid)
    sub_dataset = torch.utils.data.Subset(text_dataset, idxs)
    # pax({
    #     "sub_dataset": sub_dataset,
    #     "sub_dataset / len": len(sub_dataset),
    #     "sub_dataset / 0": sub_dataset[0],
    # })
    return sub_dataset


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


def get_valid_embeddings():
    valid_dataset = get_valid_dataset()
    embedders = get_embedders()
    embeddings = {
        m : e.embed_text_dataset(valid_dataset)
        for m, e in embedders.items()
    }
    # pax({"embeddings": embeddings})
    return embeddings


# def get_indexes():

#     embeddings = get_embeddings()

#     indexes = defaultdict(dict)
#     for model_key in embeddings:
#         for index_key, index_info in index_infos.items():

#             index_path = os.path.join(
#                 get_root_dir(),
#                 "index",
#                 "%s_%s.faiss_index" % (model_key, index_key),
#             )
#             if not os.path.exists(index_path):

#                 index = faiss.index_factory(1024, index_info["name"])
#                 data = embeddings[model_key]["train"]

#                 print("%s, %s ... train index." % (model_key, index_key))
#                 index.train(data)

#                 print("%s, %s ... add to index." % (model_key, index_key))
#                 index.add(data)

#                 os.makedirs(os.path.dirname(index_path), exist_ok = True)
#                 faiss.write_index(index, index_path)

#             indexes[model_key][index_key] = {
#                 **index_info,
#                 "index" : faiss.read_index(index_path, faiss.IO_FLAG_MMAP),
#             }

#     pax(indexes)

#     return indexes
def get_indexes():
    indexes = defaultdict(dict)
    for model_key in "megatron", "huggingface":
        print("read index '%s'." % model_key)
        path = os.path.join(get_root_dir(), "index", f"{model_key}_approx.faissindex")
        indexes[model_key]["approx"] = faiss.read_index(path, faiss.IO_FLAG_MMAP)
        # indexes[model_key]["approx"] = faiss.read_index(path)
        print("read > done.")
    # pax({k:i["approx"] for k,i in indexes.items()})
    return indexes


def get_nbrs():

    nbr_path = os.path.join(get_root_dir(), "nbrs-%d.json" % n_valid)
    if not os.path.exists(nbr_path):

        embeddings = get_valid_embeddings()
        indexes = get_indexes()
        # pax({"valid_embeddings": valid_embeddings, "indexes": indexes})

        nbrs = defaultdict(dict)
        for model_key in indexes:
            for index_key, index in indexes[model_key].items():

                pax({
                    "model_key" : model_key,
                    "index_key" : index_key,
                })

                if index_key == "approx":
                    search_params = {
                        "efSearch" : 16, # args.retro_ef_search,
                        "nprobe" : 4096, # args.retro_nprobe,
                    }
                for k, p in search_params.items():
                    faiss.ParameterSpace().set_index_parameter(index, k, p)

                print("search %s, %s." % (model_key, index_key))
                _, _nbrs = index.search(embeddings[model_key], max_nbrs)
                nbrs[model_key][index_key] = _nbrs.tolist()

        pax({
            **{f"emb/{k}":e.tolist() for k,e in embeddings.items()},
            **{f"nbr/{k}":d["approx"] for k,d in nbrs.items()},
        })
        with open(nbr_path, "w") as f:
            json.dump(nbrs, f)

    with open(nbr_path) as f:
        nbrs = json.load(f)
        for m in nbrs:
            for i in nbrs[m]:
                nbrs[m][i] = np.array(nbrs[m][i]).astype("i8")

    pax(nbrs)

    return nbrs


def get_acc():

    acc_path = os.path.join(get_root_dir(), "accs-%d.json" % n_valid)
    if not os.path.exists(acc_path):

        nbrs = get_nbrs()
        # pax({"nbrs": nbrs})

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

        with open(acc_path, "w") as f:
            json.dump(accs, f)

    with open(acc_path) as f:
        accs = json.load(f)

    # pax(accs)

    return accs


# >>>
def test_megatron_index(index):

    args = get_retro_args()

    faiss.ParameterSpace().set_index_parameter(index, "efSearch", 16)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", 4096)

    embedder = BertEmbedder(
        args.retro_bert_batch_size,
        args.retro_bert_max_chunk_length,
        force_megatron = True,
    )
    dataset = get_merged_valid_dataset()
    dataset = GPTToTextDataset(dataset)
    dataset = torch.utils.data.Subset(dataset, range(10))
    embeddings = embedder.embed_text_dataset(dataset)

    _, nbrs = index.search(embeddings, max_nbrs)

    pax({
        "embedder" : embedder,
        "embeddings" : embeddings.tolist(),
        "nbrs" : nbrs.tolist(),
    })


def run_megatron_test_v0():

    index = faiss.read_index("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/added.faissindex", faiss.IO_FLAG_MMAP)

    test_megatron_index(index)


def run_megatron_test_v1():

    def load_block(path):
        with h5py.File(path) as f:
            return np.copy(f["data"])
    def load_codes(index, path):
        with h5py.File(path) as f:
            return index.sa_encode(np.copy(f["data"]))

    import concurrent
    import glob
    import h5py

    index = faiss.read_index("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/empty.faissindex")
    index_ivf = faiss.extract_index_ivf(index)
    train_paths = sorted(glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/train_tmp/*.hdf5"))
    # >>>
    # train_paths = train_paths[:20]
    # train_paths = train_paths[:50]
    train_paths = train_paths[:200]
    # <<<

    # pax({"train_path_start_idxs": train_path_start_idxs})
    # for train_path in tqdm(train_paths):
    #     with h5py.File(train_path) as f:
    #         index.add(np.copy(f["data"]))

    # >>>
    # bs = 30
    # with concurrent.futures.ThreadPoolExecutor(max_workers = bs) as executor:

    #     train_path_start_idxs = list(range(0, len(train_paths), bs))
    #     for train_path_start_idx in tqdm(train_path_start_idxs):
    #         train_path_end_idx = min(len(train_paths), train_path_start_idx + bs)

    #         # Launch threads to load block data.
    #         futures = []
    #         for tpi in range(train_path_start_idx, train_path_end_idx):
    #             futures.append(executor.submit(load_block, train_paths[tpi]))

    #         # Collect block data.
    #         block_datas = []
    #         for future in futures:
    #             block_datas.append(future.result())

    #         # Concatenate blocks.
    #         group_data = np.concatenate(block_datas, axis = 0)

    #         # pax({"group_data": group_data})

    #         index.add(group_data)
    # +++
    bs = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers = bs) as executor:

        train_path_start_idxs = list(range(0, len(train_paths), bs))
        for train_path_start_idx in tqdm(train_path_start_idxs):
            train_path_end_idx = min(len(train_paths), train_path_start_idx + bs)

            # Launch threads to load block data.
            futures = []
            for tpi in range(train_path_start_idx, train_path_end_idx):
                futures.append(executor.submit(
                    load_codes,
                    index,
                    # index_ivf,
                    train_paths[tpi],
                ))

            # Collect block data.
            block_codes = []
            for future in futures:
                block_codes.append(future.result())

            # Concatenate blocks.
            group_codes = np.concatenate(block_codes, axis = 0)

            # pax({"block_codes": block_codes, "group_codes": group_codes})

            # index.add_sa_codes(group_codes)
            index_ivf.add_sa_codes(group_codes)
    # <<<

    test_megatron_index(index)

    pax({
        "train_paths" : train_paths,
        "index" : index,
    })
    
# <<<


def run_bert_comparison():

    # >>>
    # run_megatron_test_v0()
    run_megatron_test_v1()
    exit()
    # <<<

    faiss.omp_set_num_threads(64)

    # indexes = get_indexes()
    acc_map = get_acc()

    pax({
        # "n_samples" : n_samples,
        # "indexes" : sorted(list(set(
        #     "%s ... %s" % (info["name"], info["search"])
        #     for imap in indexes.values()
        #     for info in imap.values()
        # ))),
        "acc_map" : acc_map,
        # "time_map" : timer.time_map,
    })
