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

from megatron import get_tokenizer
from megatron.data.bert_dataset import build_training_sample
# from megatron.tokenizer.gpt2_tokenization import GPT2Tokenizer
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)

# >>>
from lutil import pax
# <<<


class GPTChunkDataset(torch.utils.data.Dataset):

    # def __init__(self, indexed_dataset, chunk_index, eods):
    #     self.indexed_dataset = indexed_dataset
    #     self.chunk_index = chunk_index
    #     self.eods = eods
    def __init__(self, indexed_datasets, dataset_offsets, chunk_index,
                 max_chunk_len):
        self.indexed_datasets = indexed_datasets
        self.dataset_offsets = dataset_offsets
        self.chunk_index = chunk_index
        self.max_gpt_chunk_len = max_chunk_len

        dataset_ids = []
        for i in range(len(dataset_offsets) - 1):
            dataset_ids.append([i] * (dataset_offsets[i+1] - dataset_offsets[i]))
        self.dataset_ids = [ i for ii in dataset_ids for i in ii ]

        # >>>
        self.gpt_tokenizer = _GPT2BPETokenizer(
            vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json",
            merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt",
        )
        # <<<

        # pax({
        #     "dataset_offsets" : self.dataset_offsets,
        #     "dataset_ids" :
        #     [ "%d / %s ..." % (len(ii), str(ii[:10])) for ii in dataset_ids ],
        #     "*dataset_ids / len" : len(self.dataset_ids),
        # })

    def __len__(self):
        # raise Exception("length?")
        # # -1 is due to data structure used to retieve the index:
        # #    sample i --> [sample_idx[i], sample_idx[i+1])
        # return self.sample_idx.shape[0] - 1
        return len(self.chunk_index)

    def __getitem__(self, chunk_id):

        # dataset_idx = self.chunk_index_to_dataset_index(chunk_idx)

        dataset_id = self.dataset_ids[chunk_id]
        doc_id, chunk_start_idx, chunk_end_idx, _ = self.chunk_index[chunk_id]
        chunk_len = chunk_end_idx - chunk_start_idx
        indexed_dataset = self.indexed_datasets[dataset_id]

        token_ids = indexed_dataset.get(doc_id,
                                        offset = chunk_start_idx,
                                        length = chunk_len)

        if chunk_len != self.max_gpt_chunk_len:
            assert chunk_len < self.max_gpt_chunk_len, "invalid chunk len."
            token_ids = token_ids.tolist()
            token_ids += [self.gpt_tokenizer.eod_id] * \
                (self.max_gpt_chunk_len - chunk_len)
            # pax({
            #     "tokenizer" : self.gpt_tokenizer,
            #     "token_ids" : "%d ... %s" % (
            #         len(token_ids),
            #         str(token_ids),
            #     ),
            # })

        # pax({
        #     # "indexed_dataset" : indexed_dataset,
        #     "chunk_id" : chunk_id,
        #     "dataset_id" : dataset_id,
        #     "doc_id" : doc_id,
        #     "chunk_start_idx" : chunk_start_idx,
        #     "chunk_end_idx" : chunk_end_idx,
        #     "chunk" : chunk,
        # })

        return {'text': np.array(token_ids, dtype=np.int64)}

class BertChunkDataset(GPTChunkDataset):

    def __init__(self, indexed_datasets, dataset_offsets,
                 chunk_index, max_chunk_len,
                 max_seq_len,
                 micro_batch_size,
                 masked_lm_prob,
                 seed,
                 binary_head,
    ):

        super().__init__(indexed_datasets, dataset_offsets,
                         chunk_index, max_chunk_len)

        # >>>
        # self.bert_tokenizer = get_tokenizer()
        self.bert_tokenizer = _BertWordPieceTokenizer(
            vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt",
            lower_case = True,
        )
        # <<<

        self.max_seq_len = max_seq_len
        self.micro_batch_size = micro_batch_size

        # Params to store.
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.binary_head = binary_head

        # Vocab stuff.
        self.vocab_id_list = list(self.bert_tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = self.bert_tokenizer.inv_vocab
        self.cls_id = self.bert_tokenizer.cls
        self.sep_id = self.bert_tokenizer.sep
        self.mask_id = self.bert_tokenizer.mask
        self.pad_id = self.bert_tokenizer.pad

        # Sort samples by bert chunk length.
        bert_chunk_lens = list(enumerate(self.chunk_index[:, 3]))
        print(" > sort / start.")
        import time
        t = time.time()
        bert_chunk_lens.sort(key = lambda item : item[1])
        # >>>
        # bert_chunk_lens.reverse() # for debugging.
        # <<<
        self.sample_idxs = [ item[0] for item in bert_chunk_lens ]
        print(" > sort / end. [ %.2f sec ]" % (time.time() - t))

        # Group samples idxs into microbatches.
        n_chunks = len(self.sample_idxs)
        self.batch_chunk_lens = []
        for batch_start_idx in range(0, n_chunks, micro_batch_size):
            batch_end_idx = min(n_chunks, batch_start_idx + micro_batch_size)
            batch_chunk_lens = [item[1].item() for item in
                                bert_chunk_lens[batch_start_idx:batch_end_idx]]
            max_chunk_len = max(batch_chunk_lens)
            self.batch_chunk_lens.append(max_chunk_len)


    def __getitem__(self, sample_id):

        chunk_id = self.sample_idxs[sample_id]

        gpt_token_ids = super().__getitem__(chunk_id)["text"]
        gpt_token_ids = [t for t in gpt_token_ids.tolist()
                         if t != self.gpt_tokenizer.eod]

        text = self.gpt_tokenizer.detokenize(gpt_token_ids)

        bert_token_ids = self.bert_tokenizer.tokenize(text)

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + chunk_id) % 2**32))

        batch_id = sample_id // self.micro_batch_size
        batch_chunk_len = min(self.max_seq_len - 2, self.batch_chunk_lens[batch_id])

        if len(bert_token_ids) > batch_chunk_len:
            bert_token_ids = bert_token_ids[:batch_chunk_len]

        return build_training_sample([bert_token_ids],
                                     len(bert_token_ids),
                                     batch_chunk_len + 2, # +2 == cls, sep
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, np_rng,
                                     self.binary_head)
