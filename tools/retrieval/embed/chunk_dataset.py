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
from megatron.tokenizer.tokenizer import _GPT2BPETokenizer

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
        self.max_chunk_len = max_chunk_len

        dataset_ids = []
        for i in range(len(dataset_offsets) - 1):
            dataset_ids.append([i] * (dataset_offsets[i+1] - dataset_offsets[i]))
        self.dataset_ids = [ i for ii in dataset_ids for i in ii ]

        # pax({
        #     "dataset_offsets" : self.dataset_offsets,
        #     "dataset_ids" :
        #     [ "%d / %s ..." % (len(ii), str(ii[:10])) for ii in dataset_ids ],
        #     "*dataset_ids / len" : len(self.dataset_ids),
        # })

    def __len__(self):
        raise Exception("length?")
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, chunk_id):

        # dataset_idx = self.chunk_index_to_dataset_index(chunk_idx)

        dataset_id = self.dataset_ids[chunk_id]
        document_id, chunk_start_idx, chunk_end_idx = self.chunk_index[chunk_id]
        chunk_len = chunk_end_idx - chunk_start_idx
        indexed_dataset = self.indexed_datasets[dataset_id]

        chunk = indexed_dataset.get(document_id,
                                    offset = chunk_start_idx,
                                    length = chunk_len)

        if chunk_len != self.max_chunk_len:
            assert chunk_len < self.max_chunk_len, "invalid chunk len."
            raise Exception("extend chunk with tokenizer.eod.")

        # pax({
        #     # "indexed_dataset" : indexed_dataset,
        #     "chunk_id" : chunk_id,
        #     "dataset_id" : dataset_id,
        #     "document_id" : document_id,
        #     "chunk_start_idx" : chunk_start_idx,
        #     "chunk_end_idx" : chunk_end_idx,
        #     "chunk" : chunk,
        # })

        return {'text': np.array(chunk, dtype=np.int64)}

class BertChunkDataset(GPTChunkDataset):

    def __init__(self, indexed_datasets, dataset_offsets,
                 chunk_index, max_chunk_len, max_embed_chunk_len,
                 # num_epochs,
                 # max_num_samples,
                 masked_lm_prob,
                 # max_seq_length,
                 # short_seq_prob,
                 seed,
                 binary_head,
    ):

        super().__init__(indexed_datasets, dataset_offsets,
                         chunk_index, max_chunk_len)

        self.max_embed_chunk_len = max_embed_chunk_len

        # >>>
        # gpt_tokenizer = GPT2Tokenizer(
        self.gpt_tokenizer = _GPT2BPETokenizer(
            vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json",
            merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt",
        )
        self.bert_tokenizer = get_tokenizer()
        # <<<

        # Params to store.
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        # self.max_seq_length = max_seq_length
        self.binary_head = binary_head

        # Vocab stuff.
        # tokenizer = get_tokenizer()
        self.vocab_id_list = list(self.bert_tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = self.bert_tokenizer.inv_vocab
        self.cls_id = self.bert_tokenizer.cls
        self.sep_id = self.bert_tokenizer.sep
        self.mask_id = self.bert_tokenizer.mask
        self.pad_id = self.bert_tokenizer.pad
        

    def __getitem__(self, chunk_id):

        gpt_token_ids = super().__getitem__(chunk_id)["text"]
        gpt_token_ids = [t for t in gpt_token_ids.tolist()
                         if t != self.gpt_tokenizer.eod]

        text = self.gpt_tokenizer.detokenize(gpt_token_ids)

        bert_token_ids = self.bert_tokenizer.tokenize(text)
        # >>>
        # bert_chunk_len = len(bert_token_ids)
        # if bert_chunk_len != self.max_chunk_len:
        #     assert bert_chunk_len < self.max_chunk_len, "invalid chunk len."
        #     bert_token_ids += [self.bert_tokenizer.eos_token_id] * \
        #         (self.max_chunk_len - bert_chunk_len)
        #     # pax({
        #     #     "bert_tokenizer" : self.bert_tokenizer,
        #     #     "bert_token_ids" : "%d ... %s" % (
        #     #         len(bert_token_ids),
        #     #         str(bert_token_ids),
        #     #     ),
        #     # })
        # +++
        # pax({
        #     "max_chunk_len" : self.max_chunk_len,
        #     "max_embed_chunk_len" : self.max_embed_chunk_len,
        # })

        # Final token will be padded in 'build_sample'.
        # assert len(bert_token_ids) <= self.max_chunk_len - 2 # cls, sep[, eos]
        assert len(bert_token_ids) <= self.max_embed_chunk_len - 2 # cls, sep[, eos]
        # <<<

        # pax({
        #     "gpt_token_ids" : gpt_token_ids,
        #     "bert_token_ids" : bert_token_ids,
        #     "gpt_token_ids / str" : str(gpt_token_ids.tolist()),
        #     "bert_token_ids / str" : str(bert_token_ids.tolist()),
        #     "text" : text,
        # })

        # >>>
        # return {'text': np.array(bert_token_ids, dtype=np.int64)}
        # <<<

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + chunk_id) % 2**32))

        # >>>
        # pax(0, {
        #     "start_idx" : int(start_idx),
        #     "end_idx" : int(end_idx),
        #     "seq_length" : int(seq_length),
        #     "indexed_dataset / %d" % start_idx : self.indexed_dataset[start_idx],
        #     "sample" : sample,
        #     "seed" : self.seed,
        #     # "seq_length" : seq_length,
        #     # "max_seq_length" : self.max_seq_length,
        #     # "vocab_id_list" : self.vocab_id_list,
        #     # "vocab_id_to_token_dict" : self.vocab_id_to_token_dict,
        #     # "cls_id" : self.cls_id,
        #     # "sep_id" : self.sep_id,
        #     # "mask_id" : self.mask_id,
        #     # "pad_id" : self.pad_id,
        #     # "masked_lm_prob" : self.masked_lm_prob,
        #     # "np_rng" : np_rng,
        #     # "binary_head" : self.binary_head,
        # })
        # <<<

        sample = build_training_sample([bert_token_ids],
                                       len(bert_token_ids), # self.max_chunk_len,
                                       self.max_embed_chunk_len,  # for padding
                                       self.vocab_id_list,
                                       self.vocab_id_to_token_dict,
                                       self.cls_id, self.sep_id,
                                       self.mask_id, self.pad_id,
                                       self.masked_lm_prob, np_rng,
                                       self.binary_head)

        # pax({"sample": sample})

        return sample
