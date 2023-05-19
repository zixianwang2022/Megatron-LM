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

import json
import collections
from multiprocessing.sharedctypes import Value
import os
import torch
import numpy as np
import glob
from megatron import get_tokenizer, get_args

def format_answer(answer):
    return " {}".format(answer)

"""GPT ft dataset."""
def preprocess(data_file, inference_only=False):

    args = get_args()
    nq_examples = []
    for my_data_file in sorted(glob.glob(data_file)):
        with open(my_data_file, "r", encoding='utf-8') as f:
            nq_examples.extend(json.load(f))
    
    data = []
    for instance in nq_examples:
        question = instance["question"]

        if inference_only:
            data.append((question, None, None))
        else:
            if "answers" in instance:
                answers = instance["answers"]
            elif "answer" in instance:
                if type(instance["answer"]) is str:
                    answers = [instance["answer"]]
                elif type(instance["answer"]) is list:
                    answers = instance["answer"]
                else:
                    answers = [str(instance["answer"])]
            else:
                raise ValueError("need to have answer or answers")
            if len(answers) < 1:
                continue
            else:
                ## only take answer 0
                if type(answers[0]) is dict:
                    answers = [answers[0]["text"].strip()]
                elif type(answers[0]) is str:
                    answers = [answers[0]]
                else:
                    raise ValueError("unsupported type for answer(s)")

            for answer in answers:
                answer = format_answer(answer)
                data.append((question, answer, None))
    
    return data

def get_processed_dataset(name, data_folder, processed=True, ratio=None, index=None, num_samples=None):

    training_file = data_folder + "/{}/{}_QA_train*.json".format(name, name)
    validation_file = data_folder + "/{}/{}_QA_dev.json".format(name, name)
    # test_file = data_folder + "/{}/{}_QA_test.json"

    dataset = {}
    dataset["train"] = preprocess(training_file)
    dataset["valid"] = preprocess(validation_file)
    dataset["test"] = preprocess(validation_file)
    
    print(name, "train", len(dataset["train"]))
    print(name, "valid", len(dataset["valid"]))
    print(name, "test", len(dataset["test"]))

    return dataset

class FtDataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, max_seq_length, 
                 max_seq_length_dec=0):

        # Params to store.
        self.dataset_name = name ## dataset_name equals to data_prefix in pretrain
        self.max_seq_length = max_seq_length

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.eos_id = tokenizer.eod
        self.pad_id = tokenizer.eod

        self.args = get_args()

        # count_stat(indexed_dataset, tokenizer)
    def __len__(self):
        return len(list(self.indexed_dataset))

    def __getitem__(self, idx):

        idx = idx % len(self.indexed_dataset)
        sample = self.indexed_dataset[idx]
       
        return build_normal_training_sample(sample,
                                self.max_seq_length,  # needed for padding
                                self.pad_id, self.eos_id,
                                self.dataset_name,
                                self.args.ft_neighbours,
                                self.args.shuffle_topn)

def build_normal_training_sample(sample,
                          max_seq_length,
                          pad_id,
                          eos_id,
                          dataset_name,
                          ft_neighbours=1,
                          shuffle_topn=False):

    # unpack tokens
    query, answer, neighbours = sample
    
    # tokenization
    tokenizer = get_tokenizer()

    input_tokens = tokenizer.tokenize(query)
    output_tokens = tokenizer.tokenize(answer)

    # print(repr(tokenizer.detokenize(input_tokens)), repr(tokenizer.detokenize(output_tokens)), dataset_name)
    # Padding
    tokens, answer_mask \
        = pad_and_convert_to_numpy(input_tokens, output_tokens,
                                   pad_id, max_seq_length, eos_id)

    train_sample = {
        'text': tokens,
        'answer_mask': answer_mask,
    }
    return train_sample


def pad_and_convert_to_numpy(input_ids, output_ids,
                             pad_id, max_seq_length, 
                             eos_id):
    """Pad sequences and convert them to numpy."""
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length - 1]

    if len(input_ids + output_ids) > max_seq_length:
        output_ids = output_ids[:max_seq_length - len(input_ids)]

    tokens = input_ids + output_ids
    answer_mask = [0] * len(input_ids) + [1] * len(output_ids)

    #padding
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0

    # Tokens.
    filler = [pad_id] * padding_length
    tokens = np.array(tokens + [eos_id] + filler, dtype=np.int64)

    # answer mask
    answer_mask = answer_mask + [1] + [0] * padding_length
    answer_mask = np.array(answer_mask, dtype=np.int64)

    return tokens, answer_mask
