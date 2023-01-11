# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import faiss
import glob
import h5py
import json
import numpy as np
import os
from tqdm import tqdm

from megatron import get_retro_args
from tools.retro.index import Index
from tools.retro.index.utils import get_index_dir

from ..acc import rowwise_intersection

max_nbrs = 200
persist_data = True # False
index_infos = {
    "exact" : {
        "name" : "Flat",
        "search" : {},
    },
    # "approx" : { # b 1 [ n 100K ]
    #     "name" : "IVF4096_HNSW8,Flat",
    #     "search" : {"efSearch" : 8, "nprobe" : 256},
    # },
    # "approx" : { # b 10 [ n 1M ]
    #     "name" : "IVF8192_HNSW8,Flat",
    #     "search" : {"efSearch" : 8, "nprobe" : 256},
    # },

    "approx" : { # b 1 [ n 100K ]
        "n_samples" : 100000,
        "name" : "IVF8192_HNSW4,Flat",
        "search" : {"efSearch" : 4, "nprobe" : 8},
    },
    # "approx" : { # b 10 [ n 1M ]
    #     "n_samples" : 1000000,
    #     "name" : "IVF65536_HNSW4,Flat", # 8
    #     "search" : {"efSearch" : 2, "nprobe" : 64}, # 4
    # },
    # "approx" : { # n 10M
    #     "n_samples" : 10000000,
    #     "name" : "IVF524288_HNSW8,Flat", # 8
    #     "search" : {"efSearch" : 4, "nprobe" : 512}, # 4
    # },
    # "approx" : { # 66M
    #     "n_samples" : 66000000,
    #     "name" : "IVF262144_HNSW32,Flat",
    #     "search" : {"efSearch" : 16, "nprobe" : 4096},
    # },
}

def get_n_samples():
    return {
        "train" : index_infos["approx"]["n_samples"],
        "valid" : 10000,
    }

def get_root_dir():
    dirname = os.path.join(
        get_index_dir(),
        "compare",
        "t%d_%s" % (get_n_samples()["train"], index_infos["approx"]["name"]),
    )
    os.makedirs(dirname, exist_ok = True)
    return dirname


def get_embeddings():

    args = get_retro_args()

    block_paths = {
        "megatron" : sorted(glob.glob("/path/to/train/embeddings/*.hdf5")),
        "huggingface" : sorted(glob.glob("/path/to/train/embeddings/*.hdf5")),
    }

    assert len(set(len(pp) for pp in block_paths.values())) == 1

    n_samples = get_n_samples()
    n_blocks = {
        k : int(np.ceil(n / args.retro_block_size))
        for k, n in n_samples.items()
    }
    block_path_ranges = {
        "train" : range(0, n_blocks["train"]),
        "valid" : range(n_blocks["train"], n_blocks["train"] + n_blocks["valid"]),
    }
    # assert block_path_ranges["valid"][-1] <= len(paths["megatron"])

    embeddings = defaultdict(dict)
    for model_key, paths in block_paths.items():
        for data_key, block_path_range in block_path_ranges.items():
            crnt_embeddings = []
            for path_idx in tqdm(
                    block_path_range,
                    "load embeds %s / %s" % (model_key, data_key),
            ):
                with h5py.File(paths[path_idx]) as f:
                    crnt_embeddings.append(np.copy(f["data"]))

            crnt_embeddings = \
                np.concatenate(crnt_embeddings, axis = 0)[:n_samples[data_key]]
            block_size = args.retro_block_size if data_key == "train" else 1000
            idxs = list(range(block_size, n_samples[data_key], block_size))
            # if idxs[-1] != n_samples[data_key]:
            #     idxs.append(n_samples[data_key])
            crnt_embeddings = np.split(crnt_embeddings, idxs, axis = 0)

            assert sum(e.shape[0] for e in crnt_embeddings) == n_samples[data_key]

            embeddings[model_key][data_key] = crnt_embeddings

    return embeddings


def get_indexes():

    embeddings = get_embeddings()
    # embeddings = None

    indexes = defaultdict(dict)
    for model_key in embeddings:
        for index_key, index_info in index_infos.items():

            index_path = os.path.join(
                get_root_dir(),
                "index",
                "%s_%s.faiss_index" % (model_key, index_key),
            )
            if not persist_data or not os.path.exists(index_path):

                embeddings = embeddings or get_embeddings()
                data = embeddings[model_key]["train"]

                index = faiss.index_factory(1024, index_info["name"])
                # faiss.ParameterSpace().set_index_parameter(index, "verbose", True)
                Index.c_verbose(index, True)

                # Move to GPU.
                if index_key == "approx":
                    index_ivf = faiss.extract_index_ivf(index)
                    clustering_index = \
                        faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
                    index_ivf.clustering_index = clustering_index
                    Index.c_verbose(index, True)
                    Index.c_verbose(index_ivf, True)
                    Index.c_verbose(index_ivf.quantizer, True)
                    Index.c_verbose(index_ivf.clustering_index, True)

                print("%s, %s ... train index." % (model_key, index_key))
                # index.train(data)
                index.train(np.concatenate(data, axis = 0))

                print("%s, %s ... add to index." % (model_key, index_key))
                # index.add(data)
                for block in tqdm(data,"add [ %s / %s ]" % (model_key,index_key)):
                    index.add(block)

                # if save_to_disk:
                os.makedirs(os.path.dirname(index_path), exist_ok = True)
                faiss.write_index(index, index_path)

            indexes[model_key][index_key] = {
                **index_info,
                "index" : faiss.read_index(index_path),
                # "index" : faiss.read_index(index_path) if save_to_disk else index,
            }

    return embeddings, indexes


def get_nbrs():

    nbr_path = os.path.join(get_root_dir(), "nbrs.json")
    if not persist_data or not os.path.exists(nbr_path):

        # embeddings = get_embeddings()
        # indexes = get_indexes()
        embeddings, indexes = get_indexes()

        nbrs = defaultdict(dict)
        for model_key in indexes:
            for index_key, index_info in indexes[model_key].items():

                index = index_info["index"]

                search_params = index_info["search"]
                for k, p in search_params.items():
                    faiss.ParameterSpace().set_index_parameter(index, k, p)

                # _, _nbrs = index.search(embeddings[model_key]["valid"],max_nbrs)
                # nbrs[model_key][index_key] = _nbrs.tolist()
                _nbrs = []
                for block in tqdm(
                        embeddings[model_key]["valid"],
                        "search [ %s / %s ]" % (model_key, index_key),
                ):
                    _nbrs.append(index.search(block, max_nbrs)[1])
                nbrs[model_key][index_key] = \
                    np.concatenate(_nbrs, axis = 0).tolist()

        with open(nbr_path, "w") as f:
            json.dump(nbrs, f)

    with open(nbr_path) as f:
        nbrs = json.load(f)
        for m in nbrs:
            for i in nbrs[m]:
                nbrs[m][i] = np.array(nbrs[m][i]).astype("i8")

    return nbrs


def get_acc():

    acc_path = os.path.join(get_root_dir(), "accs.json")
    if not persist_data or not os.path.exists(acc_path):

        nbrs = get_nbrs()

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

    return accs


def compare_bert_partial_db():

    acc_map = get_acc()

    print(n_samples)
    print(indexes)
    print(acc_map)
    exit()
