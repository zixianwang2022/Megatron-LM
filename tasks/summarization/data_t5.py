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

"""Summarization dataset."""

import random
from abc import ABC
from abc import abstractmethod

from torch.utils.data import Dataset

from megatron import print_rank_0, get_args
from tasks.data_utils_t5 import build_sample
from tasks.data_utils_t5 import build_tokens_types_paddings_from_text


class SummarizationAbstractDataset(ABC, Dataset):
    """Summarization base dataset class."""

    def __init__(self, task_name, dataset_name, datapaths,
                 tokenizer, max_seq_length, decoder_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.decoder_seq_length = decoder_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        # Process the files.
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)
        src_path, trg_path = datapaths[0], datapaths[1]
        self.samples = self.process_samples_from_paths(src_path,
                                                       trg_path)
        args = get_args()
        if args.sample_rate < 1:  # subsample
            k = int(len(self.samples) * args.sample_rate)
            self.samples = random.sample(self.samples, k)

        print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        enc_ids, tokentypes_enc, dec_in_ids, \
        dec_out_ids, loss_mask = \
            build_tokens_types_paddings_from_text(
                raw_sample['source'],
                raw_sample['target'],
                self.tokenizer,
                self.max_seq_length)
        sample = build_sample(enc_ids,
                              tokentypes_enc,
                              dec_in_ids,
                              dec_out_ids,
                              loss_mask)
        return sample

    @staticmethod
    @abstractmethod
    def process_samples_from_paths(src_filename, trg_filename):
        """Abstract method that takes source and target filenames and
        returns a list of dataset samples, each sample being a dict of
            {'text': string, 'text': string}
        """
        pass
