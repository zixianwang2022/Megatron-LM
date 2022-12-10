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

from .test_old_new import test_old_new


def print_pretraining_neighbors():

    # >>>
    test_old_new()
    raise Exception("hi.")
    # <<<

    for ds in (train_ds, valid_ds):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # for sample_idx in range(0, len(ds), len(ds) // 50):
        for sample_idx in range(0, len(ds), len(ds) // 10):

            chunk_index = np.random.randint(ds.n_chunks_per_seq)

            header = "################# sample %d, chunk %d. #################" % (
                sample_idx,
                chunk_index,
            )
            print("#" * len(header))
            print(header)
            print("#" * len(header))

            # chunk_idxs = list(range(
            #     sample_idx * ds.n_chunks_per_seq,
            #     (sample_idx + 1) * ds.n_chunks_per_seq,
            # ))

            sample = ds[sample_idx]
            seq_token_ids = sample["text"].tolist()

            # Iterate chunks.
            chunk_length = retro_args.retro_gpt_chunk_length
            # for chunk_index in range(ds.n_chunks_per_seq):
            chunk_token_ids = seq_token_ids \
                [(chunk_index * chunk_length):((chunk_index + 1) * chunk_length)]

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print_tokens("CHUNK", chunk_token_ids)

            for nbr_index, retrieved_token_ids in \
                enumerate(sample["neighbor_tokens"][chunk_index]):

                nbr_token_ids = retrieved_token_ids[:chunk_length]
                cnt_token_ids = retrieved_token_ids[chunk_length:]
                print()
                print_tokens("NBR", nbr_token_ids)
                print_tokens("CNT", cnt_token_ids)
