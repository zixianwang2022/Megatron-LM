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
import glob
import h5py
import numpy as np
import os
import torch

from megatron import get_args
from tools.retro.db.dataset import \
    get_gpt_chunk_dataset_map as get_db_gpt_chunk_dataset_map
from tools.retro.pretraining.chunk_dataset import \
    get_gpt_chunk_dataset_map as get_pretraining_gpt_chunk_dataset_map
from tools.retro.utils import get_num_chunks_per_seq

# >>>
from lutil import pax
# <<<


class IdPathMap:

    def __init__(self, paths):
        # assert isinstance(paths, list)
        self.paths = paths
        self.path_index_map = {p:i for i,p in enumerate(paths)}
        self.id_index_map = {}


    def __str__(self):
        return "%d paths; %d ids" % (len(self.paths), len(self.id_index_map))


    def add(self, id, path):
        # assert isinstance(path, str)
        self.id_index_map[id] = self.path_index_map[path]


    def __contains__(self, idx):
        return idx in self.id_index_map


    def __getitem__(self, idx):
        return self.paths[self.id_index_map[idx]]


class RetroDataset(torch.utils.data.Dataset):

    def __init__(self,
                 n_pretraining_nbrs,
                 block_size,
                 db_embed_path_map,
                 pretraining_seq_dataset,
                 pretraining_nbr_path_map,
                 pretraining_valid_seq_idxs):

        super().__init__()

        self.n_pretraining_nbrs = n_pretraining_nbrs
        self.block_size = block_size
        self.n_chunks_per_seq = get_num_chunks_per_seq()
        self.db_embed_path_map = db_embed_path_map
        self.pretraining_seq_dataset = pretraining_seq_dataset
        self.pretraining_nbr_path_map = pretraining_nbr_path_map
        self.pretraining_valid_seq_idxs = pretraining_valid_seq_idxs


    def __len__(self):
        return len(self.pretraining_valid_seq_idxs)


    def __getitem__(self, sample_idx):

        sample = self.pretraining_seq_dataset[sample_idx]

        chunk_idxs = list(range(
            sample_idx * self.n_chunks_per_seq,
            (sample_idx + 1) * self.n_chunks_per_seq,
        ))

        chunk_nbr_embeds = []
        for chunk_idx in chunk_idxs:

            # DB neighbor ids.
            nbr_path = self.pretraining_nbr_path_map[chunk_idx]
            f = h5py.File(nbr_path, "r")
            db_nbr_chunk_ids = f["neighbors"][chunk_idx % self.block_size, :self.n_pretraining_nbrs].tolist()
            f.close()

            # DB neighbor embeds.
            db_nbr_embeds = []
            for db_nbr_chunk_id in db_nbr_chunk_ids:

                # >>>
                # ni = db_nbr_chunk_id
                # ci = db_nbr_chunk_id + 1
                # np = self.db_embed_path_map[ni]
                # cp = self.db_embed_path_map[ci]
                # pax(0, {
                #     "ni" : ni,
                #     "ci" : ci,
                #     "np" : np,
                #     "cp" : cp,
                #     "ni?" : ni in self.db_embed_path_map,
                # })
                # <<<

                # Neighbor + continuation embed paths.
                db_nbr_cont_ids = db_nbr_chunk_id, db_nbr_chunk_id + 1
                # db_nbr_cont_embed_paths = [
                #     self.db_embed_path_map[ci]
                #     if ci in self.db_embed_path_map else None
                #     for ci in db_nbr_cont_ids]
                db_nbr_cont_embeds = []
                for ci in db_nbr_cont_ids:
                    if ci in self.db_embed_path_map:
                        # pax(0, {"ci": ci})
                        embed_path = self.db_embed_path_map[ci]
                        f = h5py.File(embed_path, "r")
                        embed = np.copy(f["data"][ci % self.block_size])
                        db_nbr_cont_embeds.append(embed)
                        f.close()
                    else:
                        db_nbr_cont_embeds.append(None)

                db_nbr_embeds.append({
                    "neighbor" : db_nbr_cont_embeds[0],
                    "continuation" : db_nbr_cont_embeds[1],
                })

                # pax(0, {
                #     "db_nbr_cont_ids" : db_nbr_cont_ids,
                #     "db_nbr_cont_embeds" : db_nbr_cont_embeds,
                # })

            chunk_nbr_embeds.append(db_nbr_embeds)

            # pax(0, {
            #     "db_nbr_chunk_ids" : db_nbr_chunk_ids,
            #     "db_nbr_embeds" : db_nbr_embeds,
            # })

        sample = {
            "text" : sample["text"],
            "neighbor_embeddings" : chunk_nbr_embeds,
            # "sample_token_ids" : sample_token_ids,
            # "chunks_idxs" : chunks_idxs,
        }

        # pax(0, {
        #     "sample_idx" : sample_idx,
        #     "sample" : sample,
        #     # "chunk_idxs" : chunk_idxs,
        #     # "chunk_nbr_embeds" : chunk_nbr_embeds,
        # })

        return sample

def path_to_chunk_idxs(path):
    return tuple([
        int(i) for i in os.path.splitext(
            os.path.basename(path))[0].split("-")])


# def get_db_embed_path_map(embed_dir):
# def get_id_path_map(_dir):
def get_chunk_path_map(_dir):

    paths = sorted(glob.glob(_dir + "/*.hdf5"))

    chunk_path_map = IdPathMap(paths)
    for path in paths:
        chunk_start_idx, chunk_end_idx = path_to_chunk_idxs(path)
        for chunk_idx in range(chunk_start_idx, chunk_end_idx):
            chunk_path_map.add(chunk_idx, path)

    # pax(0, {
    #     "_dir" : _dir,
    #     "paths" : paths,
    #     "chunk_path_map" : chunk_path_map,
    # })

    return chunk_path_map


# def get_valid_pretraining_seq_ids(nbr_dir, block_size):
# def get_valid_pretraining_seq_ids(nbr_dir):

#     args = get_args()

#     n_chunks_per_seq = args.retro_gpt_seq_length // args.retro_gpt_chunk_length
#     # pax(0, {"n_chunks_per_seq": n_chunks_per_seq})
#     def chunk_to_seq_idx(global_chunk_idx):
#         global_seq_idx = global_chunk_idx // n_chunks_per_seq
#         local_chunk_idx = global_chunk_idx % n_chunks_per_seq
#         # pax(0, {
#         #     "global_chunk_idx" : global_chunk_idx,
#         #     "global_seq_idx" : global_seq_idx,
#         #     "local_chunk_idx" : local_chunk_idx,
#         # })
#         return global_seq_idx, local_chunk_idx

#     nbr_paths = sorted(glob.glob(nbr_dir + "/*.hdf5"))
#     valid_seq_ids = []
#     # block_size = args
#     for nbr_path in nbr_paths:
#         global_chunk_start_idx, global_chunk_end_idx = [
#             int(i) for i in os.path.splitext(
#                 os.path.basename(nbr_path))[0].split("-")]
#         global_seq_start_idx, local_chunk_start_idx = \
#             chunk_to_seq_idx(global_chunk_start_idx)
#         global_seq_end_idx, local_chunk_end_idx = \
#             chunk_to_seq_idx(global_chunk_end_idx)

#         valid_seq_start_idx = global_seq_start_idx \
#             if local_chunk_start_idx == 0 else global_seq_start_idx + 1
#         valid_seq_end_idx = global_seq_end_idx \
#             if local_chunk_end_idx == 0 else global_seq_start_idx + 1

#         pax(0, {
#             "global start idx" : "%d (%d, %d)" % (
#                 global_chunk_start_idx,
#                 global_seq_start_idx,
#                 local_chunk_start_idx,
#             ),
#             "global end idx" : "%d (%d, %d)" % (
#                 global_chunk_end_idx,
#                 global_seq_end_idx,
#                 local_chunk_end_idx,
#             ),
#         })


#     pax(0, {
#         "nbr_paths" : nbr_paths,
#     })

#     return valid_seq_ids
def get_valid_pretraining_seq_idxs(nbr_dir):

    args = get_args()

    nbr_paths = sorted(glob.glob(nbr_dir + "/*.hdf5"))
    # n_chunks_per_seq = args.retro_gpt_seq_length // args.retro_gpt_chunk_length
    n_chunks_per_seq = get_num_chunks_per_seq()
    seq_chunk_count_map = defaultdict(lambda : 0)
    for nbr_path_index, nbr_path in enumerate(nbr_paths):
        chunk_start_idx, chunk_end_idx = path_to_chunk_idxs(nbr_path)

        for chunk_idx in range(chunk_start_idx, chunk_end_idx):
            seq_idx = chunk_idx // n_chunks_per_seq
            seq_chunk_count_map[seq_idx] += 1

    valid_seq_idxs = sorted([
        seq_idx
        for seq_idx, chunk_count in seq_chunk_count_map.items()
        if chunk_count == n_chunks_per_seq
    ])

    # pax(0, {
    #     "nbr_paths" : nbr_paths,
    #     # "seq_chunk_count_map" : seq_chunk_count_map,
    #     "valid_seq_idxs" : "%d / %s" % (len(valid_seq_idxs), str(valid_seq_idxs)),
    # })

    return valid_seq_idxs


def test_retro_dataset(args, timer):

    # Load chunk db.
    # cls.db_indexed_dataset_infos = get_db_indexed_dataset_infos(cls.args)
    # db_chunk_dataset = get_db_gpt_chunk_dataset_map()["full"]["data"]
    db_embed_dir = get_db_gpt_chunk_dataset_map()["full"]["embed_dir"]
    db_embed_path_map = get_chunk_path_map(db_embed_dir)

    # Load gpt dataset & nbrs.
    pretraining_chunk_dataset_info = \
        get_pretraining_gpt_chunk_dataset_map()["train"]
    pretraining_seq_dataset = pretraining_chunk_dataset_info["data"].seq_dataset
    pretraining_nbr_dir = pretraining_chunk_dataset_info["nbr_dir"]
    pretraining_nbr_path_map = get_chunk_path_map(pretraining_nbr_dir)
    pretraining_valid_seq_idxs = \
        get_valid_pretraining_seq_idxs(pretraining_nbr_dir)

    retro_dataset = RetroDataset(
        n_pretraining_nbrs = args.retro_nnbrs_pretraining,
        block_size = args.retro_block_size,
        db_embed_path_map = db_embed_path_map,
        pretraining_seq_dataset = pretraining_seq_dataset,
        pretraining_nbr_path_map = pretraining_nbr_path_map,
        pretraining_valid_seq_idxs = pretraining_valid_seq_idxs,
    )

    # >>>
    n_samples = 3
    samples = []
    for sample_idx in range(0, len(retro_dataset), len(retro_dataset)//n_samples):
        samples.append(retro_dataset[sample_idx])
    pax(0, {"samples": samples})
    # <<<

    pax(0, {
        "db_embed_dir" : db_embed_dir,
        "db_embed_path_map" : db_embed_path_map,
        "pretraining_seq_dataset" : pretraining_seq_dataset,
        "pretraining_nbr_dir" : pretraining_nbr_dir,
        "pretraining_nbr_path_map" : pretraining_nbr_path_map,
        "pretraining_valid_seq_idxs" : "%d / %s" % (
            len(pretraining_valid_seq_idxs),
            str(pretraining_valid_seq_idxs),
        ),
        "retro_dataset" : retro_dataset,
        "retro_dataset / len" : len(retro_dataset),
    })
