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

from megatron import get_args, get_retro_args
from tools.retro.db.utils import get_merged_train_dataset as get_db_dataset
from tools.retro.pretraining.chunk_dataset import get_chunk_dataset_map


class IdPathMap:
    '''Maps indexes to the containing block path.

    This class optimizing the mapping of a large number of indexes to the
    path of its containing block. For example, with block_size 1M, this class
    stores 1/1M as many (long) path strings, saving memory.
    '''

    def __init__(self, paths):
        self.paths = paths
        self.path_index_map = {p:i for i,p in enumerate(paths)}
        self.id_index_map = {}


    def __str__(self):
        return "%d paths; %d ids" % (len(self.paths), len(self.id_index_map))


    def add(self, id, path):
        '''Map index to a path.'''
        self.id_index_map[id] = self.path_index_map[path]


    def __contains__(self, idx):
        '''Index added to this object?'''
        return idx in self.id_index_map


    def __getitem__(self, idx):
        '''Get path from index.'''
        return self.paths[self.id_index_map[idx]]


class RetroDataset(torch.utils.data.Dataset):
    '''Dataset of retro samples.

    Each sample contains the original GPT sample, along with the token IDs
    of each neighbor of each chunk within the sequence. Neighbor array has
    shape (num_chunks_per_sample, num_neighbors, num_retrieved_tokens).
    '''

    def __init__(self,
                 n_nbrs,
                 block_size,
                 db_dataset,
                 chunk_dataset,
                 nbr_path_map):
        '''Note: chunk dataset wraps original GPT dataset (see
        chunk_dataset.py).'''

        super().__init__()

        self.n_nbrs = n_nbrs
        self.block_size = block_size
        self.db_dataset = db_dataset
        self.chunk_dataset = chunk_dataset
        self.nbr_path_map = nbr_path_map


    def __len__(self):
        return len(self.chunk_dataset.sample_dataset)


    def __getitem__(self, sample_idx):

        n_chunks_per_sample = self.chunk_dataset.n_chunks_per_sample

        # Get standard sample.
        sample = self.chunk_dataset.sample_dataset[sample_idx]

        # Sample idx to chunk idxs.
        chunk_idxs = list(range(
            sample_idx * n_chunks_per_sample,
            (sample_idx + 1) * n_chunks_per_sample,
        ))
        
        # Collect retrieved tokens.
        all_retrieved_chunk_ids = []
        all_retrieved_token_ids = []
        for chunk_idx in chunk_idxs:

            # Neighbor chunk ids.
            nbr_path = self.nbr_path_map[chunk_idx]
            with h5py.File(nbr_path, "r") as f:
                nbr_chunk_ids = f["neighbors"] \
                    [chunk_idx % self.block_size, :self.n_nbrs].tolist()

            # Retrieved (neighbor + continuation) token ids.
            retrieved_chunk_ids = []
            retrieved_token_ids = []
            for nbr_chunk_id in nbr_chunk_ids:
                current_chunk_ids = \
                    nbr_chunk_id, (nbr_chunk_id + 1) % len(self.db_dataset)
                current_token_ids = [self.db_dataset[ci]["text"]
                                     for ci in current_chunk_ids]
                retrieved_chunk_ids.append(current_chunk_ids)
                retrieved_token_ids.append(current_token_ids)

            # Collect retrieved tokens.
            all_retrieved_chunk_ids.append(retrieved_chunk_ids)
            all_retrieved_token_ids.append(retrieved_token_ids)

        # Reshape retrieved tokens.
        all_retrieved_chunk_ids = np.array(all_retrieved_chunk_ids) \
            .reshape((n_chunks_per_sample, self.n_nbrs, -1))
        all_retrieved_token_ids = np.array(all_retrieved_token_ids) \
            .reshape((n_chunks_per_sample, self.n_nbrs, -1))

        # Sample.
        sample = {
            **sample,
            "neighbor_chunks" : all_retrieved_chunk_ids, # for debugging.
            "neighbor_tokens" : all_retrieved_token_ids,
        }

        return sample


def path_to_chunk_idxs(path):
    '''Parse start/end indexes from block path name (e.g., 00010-00011.hdf5 ->
    (10, 11).'''
    return tuple([
        int(i) for i in os.path.splitext(
            os.path.basename(path))[0].split("-")])


def get_chunk_path_map(_dir):
    '''Map chunk indexes to neighbor block path (on disk).'''

    paths = sorted(glob.glob(_dir + "/*.hdf5"))

    # Build id-path map.
    chunk_path_map = IdPathMap(paths)
    for path in paths:
        chunk_start_idx, chunk_end_idx = path_to_chunk_idxs(path)
        for chunk_idx in range(chunk_start_idx, chunk_end_idx):
            chunk_path_map.add(chunk_idx, path)

    return chunk_path_map


def get_retro_datasets():
    '''Get train, valid, test retro datasets.'''

    args = get_args()
    retro_args = get_retro_args()

    # DB dataset.
    db_dataset = get_db_dataset()

    # Retro datasets.
    chunk_ds_info_map = get_chunk_dataset_map()
    retro_dataset_map = {}
    for data_key, chunk_ds_info in chunk_ds_info_map.items():

        chunk_dataset = chunk_ds_info["data"]
        nbr_dir = chunk_ds_info["nbr_dir"]
        nbr_path_map = get_chunk_path_map(nbr_dir)

        # Verify dataset prefixes.
        sample_prefix = chunk_dataset.sample_dataset.datasets[0].index_prefix
        nbr_prefix = os.path.basename(nbr_dir)
        assert sample_prefix == nbr_prefix, \
            "inconsistent dataset source; '%s' vs. '%s'." % \
            (sample_prefix, nbr_prefix)

        # Verify num chunks.
        n_sample_chunks = len(chunk_dataset)
        n_nbr_chunks = len(nbr_path_map.id_index_map)
        try:
            assert n_sample_chunks == n_nbr_chunks, \
                "inconsistent n_chunks; %d vs. %d." % \
                (n_sample_chunks, n_nbr_chunks)
        except Exception as e:
            print("nbr_dir : %s" % nbr_dir)
            print("nbr_path_map : %s" % nbr_path_map)
            raise e

        # Retro dataset.
        retro_dataset_map[data_key] = RetroDataset(
            n_nbrs = args.retro_nnbrs,
            block_size = retro_args.retro_block_size,
            db_dataset = db_dataset,
            chunk_dataset = chunk_dataset,
            nbr_path_map = nbr_path_map,
        )

    # Extract datasets.
    train_ds = retro_dataset_map.get("train", None)
    valid_ds = retro_dataset_map.get("valid", None)
    test_ds = retro_dataset_map.get("test", None)

    return train_ds, valid_ds, test_ds
