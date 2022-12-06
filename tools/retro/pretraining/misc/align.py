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
import json
import os
import pickle

from ..utils import get_pretraining_workdir

# >>>
from lutil import pax
# <<<


def get_pickle_hash(value):
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def get_seq_hashes(filename, seq_iter, get_token_id_list):

    path = os.path.join(
        get_pretraining_workdir(),
        "compare_nbrs",
        filename + ".json",
    )

    if os.path.exists(path):
        print("loading '%s'." % filename)
        with open(path) as f:
            return json.load(f)

    os.makedirs(os.path.dirname(path), exist_ok = True)

    hashes = []
    for seq_id, seq in enumerate(tqdm(seq_iter, "seq hashes / %s" % filename)):
        token_ids = get_token_id_list(seq)
        # pax(0, {
        #     "seq_id" : seq_id,
        #     "seq" : seq,
        #     "token_ids" : token_ids,
        # })
        hashes.append(get_pickle_hash(token_ids))

    with open(path, "w") as f:
        json.dump(hashes, f)

    # pax(0, {"hashes[:10]" : hashes[:10]})

    return hashes


def align_db_idxs(old_chunks, new_chunk_ds):

    # pax(0, {
    #     "old_chunks" : old_chunks,
    #     "new_chunk_ds" : new_chunk_ds,
    # })

    old_hashes = get_seq_hashes(
        "old_db_hashes",
        old_chunks,
        lambda token_ids : token_ids.tolist(),
    )
    new_hashes = get_seq_hashes(
        "new_db_hashes",
        new_chunk_ds,
        lambda sample : sample["text"].tolist(),
    )

    print("re-map db.")
    old_hash_map = { h:i for i,h in enumerate(old_hashes) }
    new_hash_map = { h:i for i,h in enumerate(new_hashes) }

    print("common db.")
    common_hashes = set(old_hashes) & set(new_hashes)

    # pax(0, {
    #     # "old_seqs" : old_seqs,
    #     # "new_seq_ds" : new_seq_ds,
    #     "old_hashes / len" : len(old_hashes),
    #     "new_hashes / len" : len(new_hashes),
    #     "old_hash_map / len" : len(old_hash_map),
    #     "new_hash_map / len" : len(new_hash_map),
    #     "common_hashes" : len(common_hashes),
    # })

    return old_hash_map, new_hash_map, list(common_hashes)


# def align_old_new_sample_idxs(old_seqs, new_seq_ds):
def align_pt_idxs(old_seqs, new_seq_ds):

    old_hashes = get_seq_hashes(
        "old_pt_hashes",
        old_seqs,
        lambda token_ids : token_ids.tolist(),
    )
    new_hashes = get_seq_hashes(
        "new_pt_hashes",
        new_seq_ds,
        lambda sample : sample["text"][:2048].tolist(),
    )

    print("re-map pt.")
    old_hash_map = { h:i for i,h in enumerate(old_hashes) }
    new_hash_map = { h:i for i,h in enumerate(new_hashes) }

    print("common pt.")
    common_hashes = set(old_hashes) & set(new_hashes)

    # pax(0, {
    #     # "old_seqs" : old_seqs,
    #     # "new_seq_ds" : new_seq_ds,
    #     "old_hashes / len" : len(old_hashes),
    #     "new_hashes / len" : len(new_hashes),
    #     "old_hash_map / len" : len(old_hash_map),
    #     "new_hash_map / len" : len(new_hash_map),
    #     "common_hashes" : len(common_hashes),
    # })

    return old_hash_map, new_hash_map, list(common_hashes)
