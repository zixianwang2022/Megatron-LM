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

from megatron import get_args, get_tokenizer
from megatron.data.bert_dataset import build_training_sample

# >>>
from lutil import pax, print_seq
# <<<


class BertEmbeddingDataset(torch.utils.data.Dataset):

    def __init__(self, text_dataset):

        super().__init__()

        args = get_args()

        # Dataset, tokenizer.
        self.text_dataset = text_dataset
        self.bert_tokenizer = get_tokenizer()

        # Params to store.
        self.max_model_seq_length = args.seq_length
        self.seed = args.seed
        self.masked_lm_prob = args.mask_prob

        # Vocab stuff.
        self.vocab_id_list = list(self.bert_tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = self.bert_tokenizer.inv_vocab
        self.cls_id = self.bert_tokenizer.cls
        self.sep_id = self.bert_tokenizer.sep
        self.mask_id = self.bert_tokenizer.mask
        self.pad_id = self.bert_tokenizer.pad


    def __len__(self):
        return len(self.text_dataset)


    def __getitem__(self, idx):

        # Text.
        text_sample = self.text_dataset[idx]
        text = text_sample["text"]

        # Bert/Wordpiece tokens (+truncate).
        bert_token_ids = self.bert_tokenizer.tokenize(text)
        bert_token_ids = bert_token_ids[:self.max_model_seq_length - 2] # cls+sep.
        if not bert_token_ids:
            bert_token_ids = [ self.bert_tokenizer.pad_id ] # hack when empty seq

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))

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
