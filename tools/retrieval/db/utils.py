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

import os

# >>>
from lutil import pax
# <<<


# def get_chunk_db_workdir(args):
def get_base_db_workdir(args):
    return os.path.join(args.retro_workdir, "db")


# def get_individual_chunk_db_path(workdir, data_name):
#     raise Exception("fix workdir.")
#     return os.path.join(workdir, f"chunk_db.{data_name}.hdf5")
def get_individual_db_path(args, data_name):
    workdir = os.path.join(get_base_db_workdir(args), "individual")
    return os.path.join(workdir, f"db.{data_name}.hdf5")


# def get_full_chunk_db_path(workdir):
#     return os.path.join(workdir, f"chunk_db.full.hdf5")
def get_full_db_info(args):
    workdir = os.path.join(get_base_db_workdir(args), "full")
    return {
        "db_path" : os.path.join(workdir, "db.hdf5"),
        "embed_dir" : os.path.join(workdir, "embed"),
    }


# def get_sampled_chunk_db_path(workdir):
#     return os.path.join(workdir, f"chunk_db.sampled.hdf5")
def get_sampled_db_info(args):
    workdir = os.path.join(get_base_db_workdir(args), "sampled")
    return {
        "db_path" : os.path.join(workdir, "db.hdf5"),
        "embed_dir" : os.path.join(workdir, "embed"),
    }


def get_db_info_map(args):
    return {
        "full" : get_full_db_info(args),
        "sampled" : get_sampled_db_info(args),
    }


# def get_chunk_embedding_path(args, data_prefix):
#     return data_prefix + f".chunk_embed_n{args.retriever_chunk_len}.hdf5"


def create_data_softlinks(data_files):

    # Soft links. [ personal space ]
    root_dir = \
        "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/preprocess/data"
    for data_index, global_file in enumerate(data_files):

        print("soft links, data %d / %d." % (data_index, len(data_files)))

        local_dir = os.path.join(
            root_dir,
            os.path.basename(os.path.dirname(global_file)),
        )
        local_prefix = os.path.join(
            local_dir,
            os.path.splitext(os.path.basename(global_file))[0],
        )
        global_prefix = os.path.splitext(global_file)[0]

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        for ext in [ "bin", "idx" ]:
            local_file = local_prefix + "." + ext
            if not os.path.exists(local_file):
                os.symlink(global_prefix + "." + ext, local_file)

        # pax(0, {
        #     "global_file" : global_file,
        #     "root_dir" : root_dir,
        #     "local_dir" : local_dir,
        #     "local_prefix" : local_prefix,
        #     "global_prefix" : global_prefix,
        # })

    pax(0, {"data_files": data_files})
    # raise Exception("soft link.")
