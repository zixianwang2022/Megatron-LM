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
import json
import os

# >>>
from lutil import pax
# <<<


def get_base_db_workdir(args):
    return os.path.join(args.retro_workdir, "db")


def get_indexed_dataset_infos_path(args):
    return os.path.join(get_base_db_workdir(args), "indexed_dataset_infos.json")


def save_indexed_dataset_infos(args, indexed_dataset_infos):
    """Save dataset order."""
    with open(get_indexed_dataset_infos_path(args), "w") as f:
        json.dump(indexed_dataset_infos, f, indent = 4)


def get_indexed_dataset_infos(args):
    path = get_indexed_dataset_infos_path(args)
    with open(path) as f:
        return json.load(f)


def get_individual_db_dir(args):
    return os.path.join(get_base_db_workdir(args), "individual")


def get_individual_db_path(args, data_name):
    return os.path.join(get_individual_db_dir(args), f"db.{data_name}.hdf5")


# def get_full_db_info(args):
#     workdir = os.path.join(get_base_db_workdir(args), "full")
#     return {
#         "db_path" : os.path.join(workdir, "db.hdf5"),
#         "embed_dir" : os.path.join(workdir, "embed"),
#         # "embed_paths" : sorted(glob.glob(embedding_dir + "/*.hdf5")),
#     }


# def get_sampled_db_info(args):
#     workdir = os.path.join(get_base_db_workdir(args), "sampled")
#     return {
#         "db_path" : os.path.join(workdir, "db.hdf5"),
#         "embed_dir" : os.path.join(workdir, "embed"),
#         # "embed_paths" : sorted(glob.glob(embedding_dir + "/*.hdf5")),
#     }


# def get_db_info_map(args):
#     return {
#         "full" : get_full_db_info(args),
#         "sampled" : get_sampled_db_info(args),
#     }
def get_db_info(args, key):
    workdir = os.path.join(get_base_db_workdir(args), key)
    db_path = os.path.join(workdir, "db.hdf5")
    embed_dir = os.path.join(workdir, "embed")
    embed_paths = sorted(glob.glob(embed_dir + "/*.hdf5")) \
        if os.path.isdir(embed_dir) else []
    return {
        "db_path" : db_path,
        "embed_dir" : embed_dir,
        "embed_paths" : embed_paths,
    }


def get_db_info_map(args):
    return {key:get_db_info(args, key) for key in ("full", "sampled")}


# def get_embedding_path_map(workdir):

#     raise Exception("move me to db/.")

#     # Directory map.
#     chunk_index_path_map = get_chunk_index_path_map(workdir)
#     embedding_path_map = {}
#     for key, chunk_index_path in chunk_index_path_map.items():

#         # Embedding sub-directory.
#         embedding_dir = os.path.join(workdir, "embed", key)
#         os.makedirs(embedding_dir, exist_ok = True)

#         # Sort data paths for reproducibility.
#         embedding_path_map[key] = {
#             "dir" : embedding_dir,
#             "data" : sorted(glob.glob(embedding_dir + "/*.hdf5")),
#         }

#     # pax(0, {
#     #     "chunk_index_path_map" : chunk_index_path_map,
#     #     "embedding_path_map" : embedding_path_map,
#     #     "workdir" : workdir,
#     # })

#     return embedding_path_map


# def get_chunk_embedding_path(args, data_prefix):
#     return data_prefix + f".chunk_embed_n{args.retriever_chunk_len}.hdf5"


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
