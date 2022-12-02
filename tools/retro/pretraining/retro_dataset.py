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

from megatron import get_retro_args
from tools.retro.db.utils import get_merged_train_dataset as get_db_chunk_dataset
from tools.retro.pretraining.chunk_dataset import \
    get_gpt_chunk_dataset_map as get_pretraining_gpt_chunk_dataset_map
# from tools.retro.utils import get_num_chunks_per_seq

# >>>
from lutil import pax
# <<<


class IdPathMap:

    def __init__(self, paths):
        self.paths = paths
        self.path_index_map = {p:i for i,p in enumerate(paths)}
        self.id_index_map = {}


    def __str__(self):
        return "%d paths; %d ids" % (len(self.paths), len(self.id_index_map))


    def add(self, id, path):
        self.id_index_map[id] = self.path_index_map[path]


    def __contains__(self, idx):
        return idx in self.id_index_map


    def __getitem__(self, idx):
        return self.paths[self.id_index_map[idx]]


# class RetroDataset(torch.utils.data.Dataset):

#     def __init__(self,
#                  db_chunk_dataset,
#                  n_nbrs,
#                  block_size,
#                  seq_dataset,
#                  nbr_path_map,
#                  legal_seq_idxs):

#         super().__init__()

#         self.db_chunk_dataset = db_chunk_dataset
#         self.n_nbrs = n_nbrs
#         self.block_size = block_size
#         self.n_chunks_per_seq = get_num_chunks_per_seq()
#         self.seq_dataset = seq_dataset
#         self.nbr_path_map = nbr_path_map
#         self.legal_seq_idxs = legal_seq_idxs

#         # >>>
#         assert len(self.legal_seq_idxs) == len(self.seq_dataset)
#         # <<<


#     def __len__(self):
#         return len(self.legal_seq_idxs)


#     def __getitem__(self, sample_idx):

#         # >>>
#         # if sample_idx >= len(self.seq_dataset):
#         #     pax(0, {
#         #         "sample_idx" : sample_idx,
#         #         "seq_dataset / len" : len(self.seq_dataset),
#         #         "legal_seq_idxs / len" : len(self.legal_seq_idxs),
#         #     })
#         # <<<

#         # Get standard sample.
#         sample_idx = self.legal_seq_idxs[sample_idx]
#         sample = self.seq_dataset[sample_idx]

#         # Sample idx to chunk idxs.
#         chunk_idxs = list(range(
#             sample_idx * self.n_chunks_per_seq,
#             (sample_idx + 1) * self.n_chunks_per_seq,
#         ))

#         # Collect retrieved tokens.
#         all_retrieved_token_ids = []
#         for chunk_idx in chunk_idxs:

#             # Neighbor chunk ids.
#             nbr_path = self.nbr_path_map[chunk_idx]
#             with h5py.File(nbr_path, "r") as f:
#                 nbr_chunk_ids = f["neighbors"] \
#                     [chunk_idx % self.block_size, :self.n_nbrs].tolist()

#             # Retrieved (neighbor + continuation) token ids.
#             retrieved_token_ids = []
#             for nbr_chunk_id in nbr_chunk_ids:
#                 current_chunk_ids = \
#                     nbr_chunk_id, (nbr_chunk_id + 1) % len(self.db_chunk_dataset)
#                 current_token_ids = [self.db_chunk_dataset[ci]["text"]
#                                      for ci in current_chunk_ids]
#                 retrieved_token_ids.append(current_token_ids)

#             # Collect retrieved tokens.
#             all_retrieved_token_ids.append(retrieved_token_ids)

#         # Reshape retrieved tokens.
#         all_retrieved_token_ids = np.array(all_retrieved_token_ids) \
#             .reshape((self.n_chunks_per_seq, self.n_nbrs, -1))

#         # Sample.
#         sample = {
#             **sample,
#             "neighbor_tokens" : all_retrieved_token_ids,
#         }

#         # pax(0, {
#         #     "all_retrieved_token_ids" : all_retrieved_token_ids,
#         #     "sample" : sample,
#         # })

#         return sample
class RetroDataset(torch.utils.data.Dataset):

    def __init__(self,
                 db_chunk_dataset,
                 n_nbrs,
                 block_size,
                 # seq_dataset,
                 chunk_dataset,
                 nbr_path_map):

        super().__init__()

        self.db_chunk_dataset = db_chunk_dataset
        self.n_nbrs = n_nbrs
        self.block_size = block_size
        # self.n_chunks_per_seq = get_num_chunks_per_seq()
        # self.seq_dataset = seq_dataset
        self.chunk_dataset = chunk_dataset
        self.nbr_path_map = nbr_path_map


    def __len__(self):
        return len(self.chunk_dataset.seq_dataset)


    def __getitem__(self, sample_idx):

        n_chunks_per_seq = self.chunk_dataset.n_chunks_per_seq

        # Get standard sample.
        sample = self.chunk_dataset.seq_dataset[sample_idx]

        # Sample idx to chunk idxs.
        chunk_idxs = list(range(
            sample_idx * n_chunks_per_seq,
            (sample_idx + 1) * n_chunks_per_seq,
        ))
        
        # Collect retrieved tokens.
        all_retrieved_token_ids = []
        for chunk_idx in chunk_idxs:

            # Neighbor chunk ids.
            nbr_path = self.nbr_path_map[chunk_idx]
            with h5py.File(nbr_path, "r") as f:
                nbr_chunk_ids = f["neighbors"] \
                    [chunk_idx % self.block_size, :self.n_nbrs].tolist()

            # Retrieved (neighbor + continuation) token ids.
            retrieved_token_ids = []
            for nbr_chunk_id in nbr_chunk_ids:
                current_chunk_ids = \
                    nbr_chunk_id, (nbr_chunk_id + 1) % len(self.db_chunk_dataset)
                current_token_ids = [self.db_chunk_dataset[ci]["text"]
                                     for ci in current_chunk_ids]
                retrieved_token_ids.append(current_token_ids)

            # Collect retrieved tokens.
            all_retrieved_token_ids.append(retrieved_token_ids)

        # Reshape retrieved tokens.
        all_retrieved_token_ids = np.array(all_retrieved_token_ids) \
            .reshape((n_chunks_per_seq, self.n_nbrs, -1))

        # Sample.
        sample = {
            **sample,
            "neighbor_tokens" : all_retrieved_token_ids,
        }

        # pax(0, {
        #     "sample" : sample,
        #     "sample_idx" : sample_idx,
        #     "chunk_idxs" : "%d / %s" % (len(chunk_idxs), str(chunk_idxs)),
        #     "all_retrieved_token_ids" : all_retrieved_token_ids,
        #     "sample" : sample,
        # })

        return sample


def path_to_chunk_idxs(path):
    return tuple([
        int(i) for i in os.path.splitext(
            os.path.basename(path))[0].split("-")])


def get_chunk_path_map(_dir):

    paths = sorted(glob.glob(_dir + "/*.hdf5"))

    chunk_path_map = IdPathMap(paths)
    for path in paths:
        chunk_start_idx, chunk_end_idx = path_to_chunk_idxs(path)
        for chunk_idx in range(chunk_start_idx, chunk_end_idx):
            chunk_path_map.add(chunk_idx, path)

    return chunk_path_map


# def get_valid_pretraining_seq_idxs(nbr_dir):
# def get_legal_seq_idxs(nbr_dir):

#     args = get_retro_args()

#     nbr_paths = sorted(glob.glob(nbr_dir + "/*.hdf5"))
#     n_chunks_per_seq = get_num_chunks_per_seq()
#     seq_chunk_count_map = defaultdict(lambda : 0)
#     for nbr_path_index, nbr_path in enumerate(nbr_paths):
#         chunk_start_idx, chunk_end_idx = path_to_chunk_idxs(nbr_path)

#         for chunk_idx in range(chunk_start_idx, chunk_end_idx):
#             seq_idx = chunk_idx // n_chunks_per_seq
#             seq_chunk_count_map[seq_idx] += 1

#     # valid_seq_idxs = sorted([
#     legal_seq_idxs = sorted([
#         seq_idx
#         for seq_idx, chunk_count in seq_chunk_count_map.items()
#         if chunk_count == n_chunks_per_seq
#     ])

#     # >>>
#     if len(seq_chunk_count_map) != len(legal_seq_idxs):
#         pax(0, {
#             "seq_chunk_count_map" : seq_chunk_count_map,
#             "legal_seq_idxs" : legal_seq_idxs,
#         })
#     # <<<

#     return legal_seq_idxs


# def get_retro_datasets():

#     args = get_retro_args()

#     # Load chunk db.
#     db_chunk_dataset = get_db_chunk_dataset()

#     # Dataset & neighbors.
#     chunk_ds_info_map = get_pretraining_gpt_chunk_dataset_map()
#     retro_dataset_map = {}
#     for data_key, chunk_ds_info in chunk_ds_info_map.items():

#         seq_dataset = chunk_ds_info["data"].seq_dataset
#         nbr_dir = chunk_ds_info["nbr_dir"]
#         nbr_path_map = get_chunk_path_map(nbr_dir)
#         legal_seq_idxs = get_legal_seq_idxs(nbr_dir)

#         raise Exception("verify seq_dataset.index_prefix == basename(nbr_dir).")
#         raise Exception("remove 'legal_seq_idxs'.")

#         # >>>
#         try:
#             assert len(legal_seq_idxs) == len(seq_dataset), \
#                 "expected %d retro samples, found %d ... did your dataset " \
#                 "arguments change?" % (len(seq_dataset), len(legal_seq_idxs))
#         except:
#             pax(0, {
#                 "db_chunk_dataset / len" : len(db_chunk_dataset),
#                 "data_key" : data_key,
#                 "chunk_ds_info" : chunk_ds_info,
#                 "seq_dataset / len" : len(seq_dataset),
#                 "nbr_dir" : nbr_dir,
#                 "nbr_path_map" : nbr_path_map,
#                 "legal_seq_idxs / len" : len(legal_seq_idxs),
#             })
#         # if len(legal_seq_idxs) > len(seq_dataset):
#         #     raise Exception("%d > %d." % (len(legal_seq_idxs), len(seq_dataset)))
#         # <<<

#         # Retro dataset.
#         retro_dataset_map[data_key] = RetroDataset(
#             db_chunk_dataset = db_chunk_dataset,
#             n_nbrs = args.retro_nnbrs_pretraining,
#             block_size = args.retro_block_size,
#             seq_dataset = seq_dataset,
#             nbr_path_map = nbr_path_map,
#             legal_seq_idxs = legal_seq_idxs,
#         )

#         # >>>
#         # pax(0, {"sample": retro_dataset[0]})
#         # <<<

#     train_ds = retro_dataset_map.get("train", None)
#     valid_ds = retro_dataset_map.get("valid", None)
#     test_ds = retro_dataset_map.get("test", None)

#     # pax(0, {
#     #     "train_ds" : train_ds,
#     #     "valid_ds" : valid_ds,
#     #     "test_ds" : test_ds,
#     #     "train_ds / len" : len(train_ds),
#     #     "valid_ds / len" : len(valid_ds),
#     #     "test_ds / len" : None if test_ds is None else len(test_ds),
#     # })

#     return train_ds, valid_ds, test_ds
def get_retro_datasets():

    args = get_retro_args()

    # Load chunk db.
    db_chunk_dataset = get_db_chunk_dataset()

    # Dataset & neighbors.
    chunk_ds_info_map = get_pretraining_gpt_chunk_dataset_map()
    retro_dataset_map = {}
    for data_key, chunk_ds_info in chunk_ds_info_map.items():

        chunk_dataset = chunk_ds_info["data"]
        # seq_dataset = chunk_dataset.seq_dataset
        nbr_dir = chunk_ds_info["nbr_dir"]
        nbr_path_map = get_chunk_path_map(nbr_dir)

        # Verify dataset prefixes.
        seq_prefix = chunk_dataset.seq_dataset.datasets[0].index_prefix
        nbr_prefix = os.path.basename(nbr_dir)
        assert seq_prefix == nbr_prefix, \
            "inconsistent dataset source; '%s' vs. '%s'." % \
            (seq_prefix, nbr_prefix)

        # Verify num chunks.
        n_seq_chunks = len(chunk_dataset)
        n_nbr_chunks = len(nbr_path_map.id_index_map)
        try:
            assert n_seq_chunks == n_nbr_chunks, \
                "inconsistent n_chunks; %d vs. %d." % (n_seq_chunks, n_nbr_chunks)
        except:
            pax(0, {
                "nbr_dir" : nbr_dir,
                "nbr_path_map" : nbr_path_map,
            })

        # Retro dataset.
        retro_dataset_map[data_key] = RetroDataset(
            db_chunk_dataset = db_chunk_dataset,
            n_nbrs = args.retro_nnbrs_pretraining,
            block_size = args.retro_block_size,
            # seq_dataset = seq_dataset,
            chunk_dataset = chunk_dataset,
            nbr_path_map = nbr_path_map,
        )

        # >>>
        # pax(0, {"sample": retro_dataset_map[data_key][0]})
        # <<<

    train_ds = retro_dataset_map.get("train", None)
    valid_ds = retro_dataset_map.get("valid", None)
    test_ds = retro_dataset_map.get("test", None)

    # pax(0, {
    #     "train_ds" : train_ds,
    #     "valid_ds" : valid_ds,
    #     "test_ds" : test_ds,
    #     "train_ds / len" : len(train_ds),
    #     "valid_ds / len" : len(valid_ds),
    #     "test_ds / len" : None if test_ds is None else len(test_ds),
    # })

    return train_ds, valid_ds, test_ds


def test_retro_dataset(timer):

    # args = get_retro_args()

    train_ds, valid_ds, _ = get_retro_datasets()

    pax(0, {})

    # >>>
    n_samples = 3
    samples = []
    for sample_idx in range(0, len(retro_dataset), len(retro_dataset)//n_samples):
        samples.append(retro_dataset[sample_idx])
    pax(0, {
        "samples" : samples,
        "samples / 0" : samples[0],
        "samples / 0 / nbrs" : samples[0]["neighbor_embeddings"],
        "samples / 0 / nbrs / 0" : samples[0]["neighbor_embeddings"][0],
        "samples / 0 / nbrs / 0 / 0" : samples[0]["neighbor_embeddings"][0][0],
    })
    # <<<

    # pax(0, {
    #     "db_embed_dir" : db_embed_dir,
    #     "db_embed_path_map" : db_embed_path_map,
    #     "pretraining_seq_dataset" : pretraining_seq_dataset,
    #     "pretraining_nbr_dir" : pretraining_nbr_dir,
    #     "pretraining_nbr_path_map" : pretraining_nbr_path_map,
    #     "pretraining_valid_seq_idxs" : "%d / %s" % (
    #         len(pretraining_valid_seq_idxs),
    #         str(pretraining_valid_seq_idxs),
    #     ),
    #     "retro_dataset" : retro_dataset,
    #     "retro_dataset / len" : len(retro_dataset),
    # })
