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

import glob
import h5py
import json
import numpy as np
import os

from megatron import get_args, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

from .dataset import GPTChunkDataset

# >>>
from lutil import pax
# <<<


def get_base_db_workdir():
    args = get_args()
    return os.path.join(args.retro_workdir, "db")


def get_indexed_dataset_infos_path():
    return os.path.join(get_base_db_workdir(), "indexed_dataset_infos.json")


def save_indexed_dataset_infos(indexed_dataset_infos):
    """Save dataset order."""
    with open(get_indexed_dataset_infos_path(), "w") as f:
        json.dump(indexed_dataset_infos, f, indent = 4)


def get_indexed_dataset_infos():
    path = get_indexed_dataset_infos_path()
    with open(path) as f:
        return json.load(f)


# def get_individual_db_info(name):
#     base_dir = os.path.join(get_base_db_workdir(), name)
#     return {
#         "db_dir" : os.path.join(base_dir, "db"),
#         "embed_dir" : os.path.join(base_dir, "embed"),
#     }
def get_individual_db_info(name):
    return {
        "db_dir" : os.path.join(get_base_db_workdir(), "individual", name, "db"),
    }


def get_individual_db(ds_id, ds_info):
    # pax(0, {"ds_id": ds_id, "ds_info": ds_info})
    db_paths = sorted(glob.glob(ds_info["db_dir"] + "/*hdf5"))
    db = np.zeros((ds_info["n_chunks_valid"], 5), dtype = "i8")
    db[:, 0] = ds_id
    start_idx = 0
    for db_path in db_paths:
        f = h5py.File(db_path, "r")
        n_chunks_current = f["chunks_valid"].shape[0]
        db[start_idx:(start_idx+n_chunks_current), 1:] = f["chunks_valid"]
        start_idx += n_chunks_current
        f.close()

    assert start_idx == ds_info["n_chunks_valid"]

    # pax(0, {"db_paths": db_paths, "ds_info": ds_info, "db": db})

    return db


# def get_db_info(key):
#     workdir = os.path.join(get_base_db_workdir(), key)
#     db_path = os.path.join(workdir, "db.hdf5")
#     embed_dir = os.path.join(workdir, "embed")
#     embed_paths = sorted(glob.glob(embed_dir + "/*.hdf5")) \
#         if os.path.isdir(embed_dir) else []
#     return {
#         "db_path" : db_path,
#         "embed_dir" : embed_dir,
#         "embed_paths" : embed_paths,
#     }


# def get_db_info_map():
#     return {key:get_db_info(key) for key in ("full", "sampled")}
# def get_blended_db_path_map():
def get_merged_db_path_map():
    base_dir = get_base_db_workdir()
    return {
        "full" : os.path.join(base_dir, "merged", "full.hdf5"),
        "sampled" : os.path.join(base_dir, "merged", "sampled.hdf5"),
    }


# def get_sampled_blended_chunk_dataset(indexed_dataset_infos):
# def get_sampled_blended_dataset(indexed_dataset_infos = None):
# def get_sampled_merged_dataset(indexed_dataset_infos = None):
def get_merged_dataset(db_type, indexed_dataset_infos = None):

    args = get_args()

    if not indexed_dataset_infos:
        indexed_dataset_infos = get_indexed_dataset_infos()

    # Build indexed datasets.
    indexed_datasets = []
    for ds_idx, ds_info in enumerate(indexed_dataset_infos):
        print_rank_0("indexed dataset %d / %d ... '%s'." %
              (ds_idx, len(indexed_dataset_infos), ds_info["name"]))
        indexed_datasets.append(make_indexed_dataset(ds_info["prefix"],
                                                     "mmap",True))

    # Load chunk db.
    db_path = get_merged_db_path_map()[db_type]
    f = h5py.File(db_path, "r")
    chunk_db = np.copy(f["chunks"])
    f.close()

    # Chunk dataset.
    chunk_dataset = GPTChunkDataset(indexed_datasets, chunk_db,
                                    args.retro_gpt_chunk_length)

    # pax(0, {
    #     "indexed_datasets" : indexed_datasets,
    #     "db_path" : db_path,
    #     "chunk_db" : chunk_db,
    #     "chunk_dataset" : chunk_dataset,
    #     "chunk_dataset / len" : len(chunk_dataset),
    #     "chunk_dataset / 0" : chunk_dataset[0],
    # })

    return chunk_dataset


def get_full_merged_dataset(indexed_dataset_infos = None):
    return get_merged_dataset("full", indexed_dataset_infos)


def get_sampled_merged_dataset(indexed_dataset_infos = None):
    return get_merged_dataset("sampled", indexed_dataset_infos)


# def create_data_softlinks(data_files):

#     # Soft links. [ personal space ]
#     root_dir = \
#         "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/preprocess/data"
#     for data_index, global_file in enumerate(data_files):

#         print("soft links, data %d / %d." % (data_index, len(data_files)))

#         local_dir = os.path.join(
#             root_dir,
#             os.path.basename(os.path.dirname(global_file)),
#         )
#         local_prefix = os.path.join(
#             local_dir,
#             os.path.splitext(os.path.basename(global_file))[0],
#         )
#         global_prefix = os.path.splitext(global_file)[0]

#         if not os.path.exists(local_dir):
#             os.mkdir(local_dir)

#         for ext in [ "bin", "idx" ]:
#             local_file = local_prefix + "." + ext
#             if not os.path.exists(local_file):
#                 os.symlink(global_prefix + "." + ext, local_file)

#         # pax(0, {
#         #     "global_file" : global_file,
#         #     "root_dir" : root_dir,
#         #     "local_dir" : local_dir,
#         #     "local_prefix" : local_prefix,
#         #     "global_prefix" : global_prefix,
#         # })

#     pax(0, {"data_files": data_files})
#     # raise Exception("soft link.")
