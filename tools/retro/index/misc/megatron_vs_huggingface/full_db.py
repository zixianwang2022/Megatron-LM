# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import faiss
import glob
import h5py
import json
import numpy as np
import os
import torch
from tqdm import tqdm

from megatron import get_retro_args
from tools.bert_embedding import BertEmbedder
from tools.retro.db.utils import get_merged_valid_dataset
from tools.retro.index.utils import get_index_dir
from tools.retro.utils import GPTToTextDataset

from ..acc import rowwise_intersection

# n_valid = 10
# n_valid = 100
# n_valid = 1000
n_valid = 10000
max_neighbors = 200
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
    return sub_dataset


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


def get_valid_embeddings():
    valid_dataset = get_valid_dataset()
    embedders = get_embedders()
    embeddings = {
        m : e.embed_text_dataset(valid_dataset)
        for m, e in embedders.items()
    }
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

#     return indexes
def get_indexes():

    index_dir = os.path.join(get_root_dir(), "index")
    os.makedirs(index_dir, exist_ok = True)

    exact_index_paths = {
        "megatron" : os.path.join(index_dir, "megatron_exact.faissindex"),
        "huggingface" : os.path.join(index_dir, "huggingface_exact.faissindex"),
    }
    embed_paths = {
        "megatron" : sorted(glob.glob("/path/to/train/embeddings/*.hdf5")),
        "huggingface" : sorted(glob.glob("/path/to/train/embeddings/*.hdf5")),
    }

    # exact_indexes = {}
    for model_key, exact_index_path in exact_index_paths.items():
        if os.path.exists(exact_index_path):
            continue
        raise Exception("re-build?")
        print("index / add.")
        exact_index = faiss.index_factory(1024, "Flat")
        for embed_path in tqdm(embed_paths[model_key], f"{model_key}/exact/add"):
            with h5py.File(embed_path) as f:
                exact_index.add(np.copy(f["data"]))
        print("index / write.")
        faiss.write_index(exact_index, exact_index_path)
        # exact_indexes[model_key] = exact_index

    index_paths = {
        "megatron" : {
            "exact" : exact_index_paths["megatron"],
            # "approx" : "/path/to/index",
        },
        "huggingface" : {
            "exact" : exact_index_paths["huggingface"],
            # "approx" : "/path/to/index",
        },
    }
    # indexes = {m:{i:faiss.read_index(p, faiss.IO_FLAG_MMAP) for i,p in ip.items()} for m,ip in index_paths.items()}
    indexes = defaultdict(dict)
    for model_key in index_paths:
        for index_key, index_path in index_paths[model_key].items():
            print("read index ... %s / %s ... %s." %
                  (model_key, index_key, index_path))
            indexes[model_key][index_key] = faiss.read_index(index_path) # , faiss.IO_FLAG_MMAP)

    return indexes


def get_neighbors():

    neighbor_path = os.path.join(get_root_dir(), "neighbors-%d.json" % n_valid)
    if not os.path.exists(neighbor_path):

        embeddings = get_valid_embeddings()
        indexes = get_indexes()

        neighbors = defaultdict(dict)
        for model_key in indexes:
            for index_key, index in indexes[model_key].items():

                if index_key == "exact":
                    search_params = {}
                elif index_key == "approx":
                    raise Exception("update for corpus.")
                    search_params = {
                        "efSearch" : 16, # args.retro_ef_search,
                        "nprobe" : 4096, # args.retro_nprobe,
                    }
                else:
                    raise Exception("specialize for '%s'." % index_key)
                for k, p in search_params.items():
                    faiss.ParameterSpace().set_index_parameter(index, k, p)

                print("search %s, %s." % (model_key, index_key))
                _, _neighbors = index.search(embeddings[model_key], max_neighbors)
                neighbors[model_key][index_key] = _neighbors.tolist()

        with open(neighbor_path, "w") as f:
            json.dump(neighbors, f)

    with open(neighbor_path) as f:
        neighbors = json.load(f)
        for m in neighbors:
            for i in neighbors[m]:
                neighbors[m][i] = np.array(neighbors[m][i]).astype("i8")

    return neighbors


def get_acc():

    acc_path = os.path.join(get_root_dir(), "accs-%d.json" % n_valid)
    if not os.path.exists(acc_path):

        neighbors = get_neighbors()

        accs = defaultdict(dict)
        for mkey0 in neighbors:
            for ikey0 in neighbors[mkey0]:
                for mkey1 in neighbors:
                    for ikey1 in neighbors[mkey1]:
                        if mkey0 == mkey1 and ikey0 == ikey1 or \
                           mkey0 < mkey1 or ikey0 < ikey1 or \
                           mkey0 != mkey1 and ikey0 != ikey1:
                            continue
                        pbar = tqdm((1, 2, 5, 10, 20, 50, 100, 200))
                        for n_neighbors in pbar:
                            pbar.set_description("acc %d" % n_neighbors)
                            if n_neighbors > max_neighbors:
                                continue
                            intsec = rowwise_intersection(
                                neighbors[mkey0][ikey0][:, :n_neighbors],
                                neighbors[mkey1][ikey1][:, :n_neighbors],
                            )
                            accs["%s/%s-%s/%s"%(mkey0,ikey0,mkey1,ikey1)][n_neighbors]\
                                = np.mean(intsec) / n_neighbors

        with open(acc_path, "w") as f:
            json.dump(accs, f)

    with open(acc_path) as f:
        accs = json.load(f)

    return accs


def compare_bert_full_db():

    faiss.omp_set_num_threads(64)

    acc_map = get_acc()

    print(acc_map)
    exit()
