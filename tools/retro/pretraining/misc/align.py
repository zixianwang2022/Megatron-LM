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

import hashlib
import h5py
import numpy as np
import os
import pickle
from tqdm import tqdm
import types

from ..utils import get_pretraining_workdir


def get_pickle_hash(value):
    return int(hashlib.sha256(pickle.dumps(value)).hexdigest()[:10], 16)


def get_sample_hashes(prefix, sample_iter, get_token_id_list):
    hash_map = {}
    for sample_id, sample in enumerate(tqdm(sample_iter, "sample hashes / %s" % prefix)):
        token_ids = get_token_id_list(sample)
        hash_map[get_pickle_hash(token_ids)] = sample_id
    return hash_map


def align_idxs(prefix, old_dict, new_dict):

    path = os.path.join(
        get_pretraining_workdir(),
        "compare_nbrs",
        prefix + ".hdf5",
    )

    if not os.path.exists(path):

        os.makedirs(os.path.dirname(path), exist_ok = True)

        print("get hashes.")
        old_hash_map = \
            get_sample_hashes(f"{prefix} / old", old_dict["data"], old_dict["post"])
        new_hash_map = \
            get_sample_hashes(f"{prefix} / new", new_dict["data"], new_dict["post"])

        print("common hashes.")
        common_hashes = list(set(old_hash_map) & set(new_hash_map))
        old_sample_ids = [ old_hash_map[h] for h in common_hashes ]
        new_sample_ids = [ new_hash_map[h] for h in common_hashes ]

        print("save hashes.")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data = np.stack(
                [old_sample_ids, new_sample_ids, common_hashes],
                axis = 1,
            ))

    print("loading '%s.hdf5'." % prefix)
    f = h5py.File(path)
    return types.SimpleNamespace(
        data = f["data"],
        # hash_map = {
        #     h.item() : i
        #     for i, (o, n, h) in enumerate(tqdm(f["data"], f"load {prefix}"))
        # },
    )


def align_db_idxs(old_db_ds, new_chunk_ds):
    return align_idxs(
        "db",
        {
            "data" : old_db_ds.chunks,
            "post" : lambda token_ids : token_ids.tolist(),
        },
        {
            "data" : new_chunk_ds,
            "post" : lambda sample : sample["text"].tolist(),
        },
    )

def align_pt_idxs(dkey, old_ds, new_ds):
    return align_idxs(
        f"pt_{dkey}",
        {
            "data" : old_ds.tokens,
            "post" : lambda tokens : tokens.tolist(),
        },
        {
            "data" : new_ds.chunk_dataset.sample_dataset,
            "post" : lambda sample : sample["text"][:2048].tolist(),
        },
    )
