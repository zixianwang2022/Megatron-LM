# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import torch

from megatron import get_tokenizer, print_rank_0
from megatron.data.bert_dataset import build_training_sample
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)

# >>>
from lutil import pax, print_seq
# <<<


class GPTChunkDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            indexed_datasets,
            dataset_ids,
            chunk_index,
            max_chunk_length,
    ):

        self.indexed_datasets = indexed_datasets
        self.dataset_ids = dataset_ids
        self.chunk_index = chunk_index
        self.max_gpt_chunk_length = max_chunk_length

        # >>>
        self.gpt_tokenizer = _GPT2BPETokenizer(
            vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json",
            merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt",
        )
        # <<<


    def __len__(self):
        return len(self.chunk_index)


    def __getitem__(self, chunk_id):

        dataset_id = self.dataset_ids[chunk_id]
        doc_id, token_start_idx, token_end_idx, _ = \
            [ value.item() for value in self.chunk_index[chunk_id] ]
        chunk_length = token_end_idx - token_start_idx
        indexed_dataset = self.indexed_datasets[dataset_id]

        token_ids = indexed_dataset.get(doc_id,
                                        offset = token_start_idx,
                                        length = chunk_length)

        if chunk_length != self.max_gpt_chunk_length:
            assert chunk_length < self.max_gpt_chunk_length, "invalid chunk len."
            token_ids = token_ids.tolist()
            token_ids += [self.gpt_tokenizer.eod_id] * \
                (self.max_gpt_chunk_length - chunk_length)

        # pax({
        #     # "indexed_dataset" : indexed_dataset,
        #     "chunk_id" : chunk_id,
        #     "dataset_id" : dataset_id,
        #     "doc_id" : doc_id,
        #     "token_start_idx" : token_start_idx,
        #     "token_end_idx" : token_end_idx,
        #     "chunk" : chunk,
        # })

        return {'text': np.array(token_ids, dtype=np.int64)}


# class BertChunkDataset(GPTChunkDataset):

#     def __init__(self,
#                  indexed_datasets,
#                  dataset_ids,
#                  chunk_index,
#                  # max_chunk_len,
#                  # max_seq_len,
#                  max_model_seq_length,
#                  # micro_batch_size,
#                  masked_lm_prob,
#                  seed,
#                  # binary_head,
#     ):

#         super().__init__(
#             indexed_datasets,
#             dataset_ids,
#             chunk_index,
#             # max_chunk_len,
#         )

#         # >>>
#         # self.bert_tokenizer = get_tokenizer()
#         self.bert_tokenizer = _BertWordPieceTokenizer(
#             vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt",
#             lower_case = True,
#         )
#         # <<<

#         self.max_model_seq_length = max_model_seq_length
#         # self.micro_batch_size = micro_batch_size

#         # Params to store.
#         self.seed = seed
#         self.masked_lm_prob = masked_lm_prob
#         # self.binary_head = binary_head

#         # Vocab stuff.
#         self.vocab_id_list = list(self.bert_tokenizer.inv_vocab.keys())
#         self.vocab_id_to_token_dict = self.bert_tokenizer.inv_vocab
#         self.cls_id = self.bert_tokenizer.cls
#         self.sep_id = self.bert_tokenizer.sep
#         self.mask_id = self.bert_tokenizer.mask
#         self.pad_id = self.bert_tokenizer.pad

#         # Sort samples by bert chunk length.
#         bert_chunk_lens = list(enumerate(self.chunk_index[:, 3]))
#         if 0:
#             print_rank_0(" > sort / start.")
#             import time
#             t = time.time()
#             # >>> [ temporarily removed. ]
#             bert_chunk_lens.sort(key = lambda item : item[1])
#             # <<<
#             # >>>
#             # bert_chunk_lens.reverse() # for debugging.
#             # <<<
#             print_rank_0(" > sort / end. [ %.2f sec ]" % (time.time() - t))

#         self.sample_idxs = [ item[0] for item in bert_chunk_lens ]

#         # >>>
#         # print_rank_0([a[1] for a in bert_chunk_lens])
#         # pax(0, {
#         #     "bert_chunk_lens / len" : len(bert_chunk_lens),
#         # })
#         # <<<

#         # Group samples idxs into microbatches.
#         n_chunks = len(self.sample_idxs)
#         self.batch_chunk_lens = []
#         for batch_start_idx in range(0, n_chunks, micro_batch_size):
#             batch_end_idx = min(n_chunks, batch_start_idx + micro_batch_size)
#             batch_chunk_lens = [item[1].item() for item in
#                                 bert_chunk_lens[batch_start_idx:batch_end_idx]]
#             max_chunk_len = max(batch_chunk_lens)
#             self.batch_chunk_lens.append(max_chunk_len)


#     def __getitem__(self, sample_id):

#         chunk_id = self.sample_idxs[sample_id]

#         # print_seq("chunk_id = %d." % chunk_id)

#         gpt_token_ids = super().__getitem__(chunk_id)["text"]
#         gpt_token_ids = [t for t in gpt_token_ids.tolist()
#                          if t != self.gpt_tokenizer.eod]

#         text = self.gpt_tokenizer.detokenize(gpt_token_ids)

#         bert_token_ids = self.bert_tokenizer.tokenize(text)

#         # Note that this rng state should be numpy and not python since
#         # python randint is inclusive whereas the numpy one is exclusive.
#         # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
#         np_rng = np.random.RandomState(seed=((self.seed + chunk_id) % 2**32))

#         batch_id = sample_id // self.micro_batch_size
#         batch_chunk_len = min(self.max_seq_len - 2, self.batch_chunk_lens[batch_id])

#         if len(bert_token_ids) > batch_chunk_len:
#             bert_token_ids = bert_token_ids[:batch_chunk_len]

#         return build_training_sample([bert_token_ids],
#                                      len(bert_token_ids),
#                                      batch_chunk_len + 2, # +2 == cls, sep
#                                      self.vocab_id_list,
#                                      self.vocab_id_to_token_dict,
#                                      self.cls_id, self.sep_id,
#                                      self.mask_id, self.pad_id,
#                                      self.masked_lm_prob, np_rng,
#                                      # add_binary_head = self.binary_head,
#                                      add_binary_head = False) # allows single sent
class BertChunkDataset(GPTChunkDataset):

    def __init__(self,
                 indexed_datasets,
                 dataset_ids,
                 chunk_index,
                 max_chunk_length,
                 max_model_seq_length,
                 masked_lm_prob,
                 seed):

        super().__init__(
            indexed_datasets,
            dataset_ids,
            chunk_index,
            max_chunk_length,
        )

        # >>>
        # self.bert_tokenizer = get_tokenizer()
        self.bert_tokenizer = _BertWordPieceTokenizer(
            vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt",
            lower_case = True,
        )
        # <<<

        self.max_model_seq_length = max_model_seq_length

        # Params to store.
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob

        # Vocab stuff.
        self.vocab_id_list = list(self.bert_tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = self.bert_tokenizer.inv_vocab
        self.cls_id = self.bert_tokenizer.cls
        self.sep_id = self.bert_tokenizer.sep
        self.mask_id = self.bert_tokenizer.mask
        self.pad_id = self.bert_tokenizer.pad


    def __getitem__(self, chunk_id):

        # GPT/BPE tokens.
        gpt_token_ids = super().__getitem__(chunk_id)["text"]
        gpt_token_ids = [t for t in gpt_token_ids.tolist()
                         if t != self.gpt_tokenizer.eod]
        text = self.gpt_tokenizer.detokenize(gpt_token_ids)

        # Bert/Wordpiece tokens (+truncate).
        bert_token_ids = self.bert_tokenizer.tokenize(text)
        bert_token_ids = bert_token_ids[:self.max_model_seq_length - 2] # cls+sep

        # pax(0, {"bert_token_ids": bert_token_ids})

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + chunk_id) % 2**32))

        # Build sample.
        sample = build_training_sample([bert_token_ids],
                                       len(bert_token_ids),
                                       len(bert_token_ids) + 2, # for cls+sep
                                       self.vocab_id_list,
                                       self.vocab_id_to_token_dict,
                                       self.cls_id, self.sep_id,
                                       self.mask_id, self.pad_id,
                                       self.masked_lm_prob, np_rng,
                                       binary_head = False)
        sample["seq_length"] = len(sample["text"])
        return sample
