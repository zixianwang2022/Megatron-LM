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


# def _print_pt_neighbors(
def print_pt_neighbors(
        gpt_tokenizer,
        embedder,
        chunk_length,
        nnbrs,
        n_chunks_per_seq,
        old_doc_ids,
        old_db_chunks,
        old_pt_seqs_train,
        old_pt_nbrs_train,
        new_pt_retro_train_ds,
        old_pt_hash_map,
        new_pt_hash_map,
        common_pt_hashes,
        old_db_hash_map,
        new_db_hash_map,
        common_db_hashes,
):

    accs = []

    # for sample_idx in range(10): # range(10, 20):
    for pt_hash_idx in range(
            0,
            len(common_pt_hashes),
            len(common_pt_hashes) // 100,
    ):

        pt_hash = common_pt_hashes[pt_hash_idx]
        old_sample_idx = old_pt_hash_map[pt_hash]
        new_sample_idx = new_pt_hash_map[pt_hash]
        sample_idxs = list(set([ old_sample_idx, new_sample_idx ]))

        # pax(0, {
        #     "pt_hash" : pt_hash,
        #     "old_sample_idx" : old_sample_idx,
        #     "new_sample_idx" : new_sample_idx,
        # })

        old_seq = old_pt_seqs_train[old_sample_idx]
        new_sample = new_pt_retro_train_ds[new_sample_idx]
        new_seq = new_sample["text"]

        # pax(0, {"old_pt_nbrs_train": old_pt_nbrs_train})

        old_nbr_ids = old_pt_nbrs_train[old_sample_idx][:, :nnbrs]
        new_nbrs = new_sample["neighbor_tokens"]
        assert nnbrs == new_nbrs.shape[1]

        chunk_idx = np.random.randint(n_chunks_per_seq)
        # for chunk_idx in range(n_chunks_per_seq):

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
        print_tokens("OLD_CHUNK", old_seq_chunk)
        print_tokens("NEW_CHUNK", new_seq_chunk)

        old_nbr_token_ids = []
        new_nbr_token_ids = []
        for nbr_idx in range(nnbrs):
            old_nbr_id = old_nbr_ids[chunk_idx][nbr_idx].item()
            old_nbr_token_ids.append(old_db_chunks[old_nbr_id])
            new_nbr_token_ids.append(new_nbrs[chunk_idx][nbr_idx][:chunk_length])

        # >>>
        # old_token_hashes = [ get_pickle_hash(ts.tolist())
        #                      for ts in old_nbr_token_ids ]
        # new_token_hashes = [ get_pickle_hash(ts.tolist())
        #                      for ts in new_nbr_token_ids ]
        # old_text_hashes = [ get_pickle_hash(gpt_tokenizer.detokenize(ts))
        #                    for ts in old_nbr_token_ids ]
        # new_text_hashes = [ get_pickle_hash(gpt_tokenizer.detokenize(ts))
        #                    for ts in new_nbr_token_ids ]
        # common_token_hashes = set(old_token_hashes) & set(new_token_hashes)
        # token_acc = len(common_token_hashes) / nnbrs
        # text_acc = len(set(old_text_hashes) & set(new_text_hashes)) / nnbrs
        # accs.append(text_acc)
        # +++
        old_nbr_hashes = [ get_pickle_hash(ts.tolist())
                           for ts in old_nbr_token_ids ]
        new_nbr_hashes = [ get_pickle_hash(ts.tolist())
                           for ts in new_nbr_token_ids ]
        common_nbr_hashes = set(old_nbr_hashes) & set(new_nbr_hashes)
        accs.append(len(common_nbr_hashes) / nnbrs)
        # +++
        # old_nbr_hashes = [ old_db_hash_map[old_nbr_ids[chunk_idx][ni].item()]
        #                    for ni in range(nnbrs)]
        # <<<

        if accs[-1] == 0.9 and old_nbr_hashes[0] not in new_nbr_hashes:
            print()
            for i, ts in enumerate(old_nbr_token_ids):
                c = old_nbr_hashes[i] in common_nbr_hashes
                print("%s : %s" % (
                    "OLD" if c else "[[OLD]]",
                    "\\n".join(gpt_tokenizer.detokenize(ts[:30]).splitlines()),
                ))
            print()
            for i, ts in enumerate(new_nbr_token_ids):
                c = new_nbr_hashes[i] in common_nbr_hashes
                print("%s : %s" % (
                    "NEW" if c else "[[NEW]]",
                    "\\n".join(gpt_tokenizer.detokenize(ts[:30]).splitlines()),
                ))

            print()
            print("ACC : %.2f." % (100 * accs[-1]))

        # >>>
        # if accs[-1] == 0.9 and old_nbr_hashes[0] not in new_nbr_hashes:
        if False:
            seq_embed = \
                embedder.embed_text(gpt_tokenizer.detokenize(old_seq_chunk))
            old_nbr_embeds = [ embedder.embed_text(gpt_tokenizer.detokenize(ts))
                               for ts in old_nbr_token_ids ]
            new_nbr_embeds = [ embedder.embed_text(gpt_tokenizer.detokenize(ts))
                               for ts in new_nbr_token_ids ]
            old_nbr_dists = [np.linalg.norm(seq_embed-e) for e in old_nbr_embeds]
            new_nbr_dists = [np.linalg.norm(seq_embed-e) for e in new_nbr_embeds]

            old_nbr_id = old_db_hash_map[old_nbr_hashes[0]]
            new_nbr_id = new_db_hash_map[old_nbr_hashes[0]]
            pax(0, {
                "old 0 hash" : old_nbr_hashes[0],
                "old 0 in old db?" : old_nbr_hashes[0] in old_db_hash_map,
                "old 0 in new db?" : old_nbr_hashes[0] in new_db_hash_map,
                "old_nbr_id" : old_nbr_id,
                "new_nbr_id" : new_nbr_id,
                "old nbr" : str(old_db_chunks[old_nbr_id]),
                "new nbr" :
                str(new_pt_retro_train_ds.db_chunk_dataset[new_nbr_id]["text"]),
                # "seq_embed" : seq_embed,
                # "old_nbr_embeds" : old_nbr_embeds,
                # "new_nbr_embeds" : new_nbr_embeds,
                "old_nbr_dists" : str(old_nbr_dists),
                "new_nbr_dists" : str(new_nbr_dists),
            })
        # <<<

    pax(0, {"acc" : np.mean(accs)})
