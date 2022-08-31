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

"""GLUE dataset."""

from abc import ABC
from abc import abstractmethod
import random

from torch.utils.data import Dataset

from megatron import print_rank_0, get_args
from tasks.t5_model_utils.data_utils import build_sample
from tasks.t5_model_utils.data_utils import mnli_build_tokens_types_paddings_from_text


class GLUEAbstractDataset(ABC, Dataset):
    """GLUE base dataset class."""

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
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_samples_from_single_path(datapath))

        args = get_args()
        if args.sample_rate < 1:  # subsample
            k = int(len(self.samples) * args.sample_rate)
            self.samples = random.sample(self.samples, k)
        print_rank_0('  >> total number of samples: {}'.format(
                    len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        enc_ids, tokentypes_enc, dec_in_ids, \
        dec_out_ids, loss_mask \
            = mnli_build_tokens_types_paddings_from_text(
            raw_sample['text_a'],
            raw_sample['text_b'],
            raw_sample['label'],
            self.tokenizer,
            self.max_seq_length,
            self.decoder_seq_length)

        sample = build_sample(enc_ids,
                              tokentypes_enc,
                              dec_in_ids,
                              dec_out_ids,
                              loss_mask)
        return sample

    @abstractmethod
    def process_samples_from_single_path(self, datapath):
        """Abstract method that takes a single path / filename and
        returns a list of dataset samples, each sample being a dict of
            {'text_a': string, 'text_b': string, 'label': int, 'uid': int}
        """
        pass
