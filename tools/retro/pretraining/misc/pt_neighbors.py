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

import numpy as np

from .align import get_pickle_hash
from .print_tokens import print_tokens

# >>>
from lutil import pax
# <<<

def print_nbrs(
        meta,
        sample_idxs,
        chunk_idx,
        old_db_ds,
        new_db_ds,
        old_sample,
        new_sample,
        db_hashes,
):

    tokenizer = meta.tokenizer
    embedder = meta.embedder
    nnbrs = meta.nnbrs
    chunk_length = meta.chunk_length
    n_chunks_per_seq = meta.n_chunks_per_seq

    old_seq = old_sample["text"][:2048]
    new_seq = new_sample["text"][:2048]
    # old_nbr_ids = old_pt_nbrs_train[old_sample_idx][:, :nnbrs]
    old_nbrs = old_sample["neighbor_tokens"][:, :, :meta.chunk_length]
    new_nbrs = new_sample["neighbor_tokens"][:, :, :meta.chunk_length]
    assert old_nbrs.shape == (n_chunks_per_seq, nnbrs, chunk_length)
    assert new_nbrs.shape == (n_chunks_per_seq, nnbrs, chunk_length)
    # assert nnbrs == new_nbrs.shape[1]

    # pax({
    #     "old_seq" : old_seq,
    #     "new_seq" : new_seq,
    #     "old_nbrs" : old_nbrs,
    #     "new_nbrs" : new_nbrs,
    # })

    old_nbr_token_ids = []
    new_nbr_token_ids = []
    for nbr_idx in range(nnbrs):
        # old_nbr_id = old_nbr_ids[chunk_idx][nbr_idx].item()
        # old_nbr_token_ids.append(old_db_chunks[old_nbr_id])
        # new_nbr_token_ids.append(new_nbrs[chunk_idx][nbr_idx][:chunk_length])
        old_nbr_token_ids.append(old_nbrs[chunk_idx][nbr_idx])
        new_nbr_token_ids.append(new_nbrs[chunk_idx][nbr_idx])

    # >>>
    # old_token_hashes = [ get_pickle_hash(ts.tolist())
    #                      for ts in old_nbr_token_ids ]
    # new_token_hashes = [ get_pickle_hash(ts.tolist())
    #                      for ts in new_nbr_token_ids ]
    # old_text_hashes = [ get_pickle_hash(tokenizer.detokenize(ts))
    #                    for ts in old_nbr_token_ids ]
    # new_text_hashes = [ get_pickle_hash(tokenizer.detokenize(ts))
    #                    for ts in new_nbr_token_ids ]
    # common_token_hashes = set(old_token_hashes) & set(new_token_hashes)
    # token_acc = len(common_token_hashes) / nnbrs
    # text_acc = len(set(old_text_hashes) & set(new_text_hashes)) / nnbrs
    # accs.append(text_acc)
    # +++
    old_nbr_hashes = [ get_pickle_hash(ts.tolist()) for ts in old_nbr_token_ids ]
    new_nbr_hashes = [ get_pickle_hash(ts.tolist()) for ts in new_nbr_token_ids ]
    common_nbr_hashes = set(old_nbr_hashes) & set(new_nbr_hashes)
    # accs.append(len(common_nbr_hashes) / nnbrs)
    acc = len(common_nbr_hashes) / nnbrs
    # +++
    # old_nbr_hashes = [ old_db_hash_map[old_nbr_ids[chunk_idx][ni].item()]
    #                    for ni in range(nnbrs)]
    # <<<

    causal = True
    # if accs[-1] == 0.9 and old_nbr_hashes[0] not in new_nbr_hashes:
    if True:
        # n_causal += 1
        causal = False

        header = "############## sample %s, chunk %d ##############" % (
            ",".join(str(i) for i in sample_idxs), chunk_idx)
        print()
        print("#" * len(header))
        print(header)
        print("#" * len(header))
        old_seq_chunk = old_seq[
            (chunk_idx * chunk_length):((chunk_idx + 1) * chunk_length)]
        new_seq_chunk = new_seq[
            (chunk_idx * chunk_length):((chunk_idx + 1) * chunk_length)]
        assert get_pickle_hash(old_seq_chunk.tolist()) == \
            get_pickle_hash(new_seq_chunk.tolist())
        # print_tokens("OLD_CHUNK", old_seq_chunk)
        # print_tokens("NEW_CHUNK", new_seq_chunk)
        print_tokens("SAMPLE", old_seq_chunk)

        print()
        for i, ts in enumerate(old_nbr_token_ids): # [:2]):
            c = old_nbr_hashes[i] in common_nbr_hashes
            print("%s : %s" % (
                "OLD" if c else "[[OLD]]",
                "\\n".join(tokenizer.detokenize(ts[:30]).splitlines()),
                # "\\n".join(tokenizer.detokenize(ts).splitlines()),
            ))
        print()
        for i, ts in enumerate(new_nbr_token_ids): # [:2]):
            c = new_nbr_hashes[i] in common_nbr_hashes
            print("%s : %s" % (
                "NEW" if c else "[[NEW]]",
                "\\n".join(tokenizer.detokenize(ts[:30]).splitlines()),
                # "\\n".join(tokenizer.detokenize(ts).splitlines()),
            ))

        print()
        print("ACC : %.2f." % (100 * acc))

    # >>>
    # if accs[-1] == 0.9 and old_nbr_hashes[0] not in new_nbr_hashes:
    # if False:
    if acc != 1:
        seq_embed = embedder.embed_text(tokenizer.detokenize(old_seq_chunk))
        old_nbr_embeds = [ embedder.embed_text(tokenizer.detokenize(ts))
                           for ts in old_nbr_token_ids ]
        new_nbr_embeds = [ embedder.embed_text(tokenizer.detokenize(ts))
                           for ts in new_nbr_token_ids ]
        old_nbr_dists = [np.linalg.norm(seq_embed-e) for e in old_nbr_embeds]
        new_nbr_dists = [np.linalg.norm(seq_embed-e) for e in new_nbr_embeds]

        try:
            diff_index = min(i for i in range(nnbrs)
                             if old_nbr_hashes[i] != new_nbr_hashes[i])
        except:
            pax({
                "old_nbr_hashes" : old_nbr_hashes,
                "new_nbr_hashes" : new_nbr_hashes,
            })
        old_nbr_id = db_hashes.old[old_nbr_hashes[diff_index]]
        new_nbr_id = db_hashes.new[new_nbr_hashes[diff_index]]
        pax(0, {
            "banned doc ids" : str(new_sample["doc_ids"]),
            "diff_index" : diff_index,
            "old diff hash" : old_nbr_hashes[diff_index],
            "old diff in old db?" : old_nbr_hashes[diff_index] in db_hashes.old,
            "old diff in new db?" : old_nbr_hashes[diff_index] in db_hashes.new,
            "old_nbr_id" : old_nbr_id,
            "new_nbr_id" : new_nbr_id,
            "old nbr" : "%d / %s" % (
                old_db_ds[old_nbr_id]["doc_id"],
                str(old_db_ds[old_nbr_id]["text"]),
            ),
            "new nbr" : "%d / %s" % (
                new_db_ds[new_nbr_id]["doc_id"],
                str(new_db_ds[new_nbr_id]["text"]),
            ),
            # "seq_embed" : seq_embed,
            # "old_nbr_embeds" : old_nbr_embeds,
            # "new_nbr_embeds" : new_nbr_embeds,
            "old_nbr_dists" : str(old_nbr_dists),
            "new_nbr_dists" : str(new_nbr_dists),
        })
    # <<<

    return acc, causal

# def _print_pt_neighbors(
# def print_pt_neighbors(
#         gpt_tokenizer,
#         embedder,
#         chunk_length,
#         nnbrs,
#         n_chunks_per_seq,
#         old_pt_ds,
#         new_pt_ds,
#         pt_hashes,
#         # db_hashes,
# ):

#     accs = []
#     n_causal = 0

#     old_pt_hash_map = pt_hashes["old"]
#     new_pt_hash_map = pt_hashes["new"]
#     common_pt_hashes = pt_hashes["common"]
#     # for sample_idx in range(10): # range(10, 20):
#     # for pt_hash_idx in range(
#     #         0,
#     #         len(common_pt_hashes),
#     #         max(1, len(common_pt_hashes) // 1000),
#     # ):
#     for rand_idx in range(100):

#         pt_hash_idx = np.random.randint(len(common_pt_hashes))

#         pt_hash = common_pt_hashes[pt_hash_idx]
#         old_sample_idx = old_pt_hash_map[pt_hash]
#         new_sample_idx = new_pt_hash_map[pt_hash]
#         sample_idxs = list(set([ old_sample_idx, new_sample_idx ]))

#         old_seq = old_pt_seqs_train[old_sample_idx]
#         new_sample = new_pt_retro_train_ds[new_sample_idx]
#         new_seq = new_sample["text"]

#         old_nbr_ids = old_pt_nbrs_train[old_sample_idx][:, :nnbrs]
#         new_nbrs = new_sample["neighbor_tokens"]
#         assert nnbrs == new_nbrs.shape[1]

#         # for chunk_idx in range(n_chunks_per_seq):
#         chunk_idx = np.random.randint(n_chunks_per_seq)

#         print_nbrs(chunk_idx)

#     # acc = np.mean(accs)
#     # causal_rate = n_causal
#     pax(0, {
#         "n_acc" : len(accs),
#         "n_causal" : n_causal,
#         "acc" : np.mean(accs),
#         "causal" : n_causal / len(accs),
#     })
def print_pt_neighbors(
        meta,
        old_pt_ds,
        new_pt_ds,
        pt_hashes,
        db_hashes,
):

    accs = []
    n_causal = 0

    # old_pt_hash_map = pt_hashes["old"]
    # new_pt_hash_map = pt_hashes["new"]
    # common_pt_hashes = pt_hashes["common"]
    for rand_idx in range(100):

        pt_hash_idx = np.random.randint(len(pt_hashes.common))

        pt_hash = pt_hashes.common[pt_hash_idx]
        old_sample_idx = pt_hashes.old[pt_hash]
        new_sample_idx = pt_hashes.new[pt_hash]
        sample_idxs = list(set([ old_sample_idx, new_sample_idx ]))

        old_sample = old_pt_ds[old_sample_idx]
        new_sample = new_pt_ds[new_sample_idx]

        # for chunk_idx in range(n_chunks_per_seq):
        chunk_idx = np.random.randint(meta.n_chunks_per_seq)

        acc, causal = print_nbrs(
            meta,
            sample_idxs,
            chunk_idx,
            old_pt_ds.db_ds,
            new_pt_ds.db_chunk_dataset,
            old_sample,
            new_sample,
            db_hashes,
        )
        accs.append(acc)
        n_causal += int(causal)

    # acc = np.mean(accs)
    # causal_rate = n_causal
    pax(0, {
        "n_acc" : len(accs),
        "n_causal" : n_causal,
        "acc" : np.mean(accs),
        "causal" : n_causal / len(accs),
    })
