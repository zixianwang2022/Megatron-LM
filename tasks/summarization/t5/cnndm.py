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

"""CNNDM dataset."""

from megatron import print_rank_0
from tasks.summarization.t5.data import SummarizationAbstractDataset


class CNNDMDataset(SummarizationAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length, decoder_seq_length):
        super().__init__('cnndm', name, datapaths, tokenizer,
                         max_seq_length, decoder_seq_length)

    @staticmethod
    def process_samples_from_paths(src_filename, trg_filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} and {} ...'.format(src_filename, trg_filename))
        samples = []
        total = 0

        with open(src_filename, 'r') as sf, open(trg_filename, 'r') as tf:
            for source, target in zip(sf, tf):
                source = source.strip()
                if len(source) == 0:
                    continue

                target = target.strip()
                if len(target) == 0:
                    continue

                sample = {'source': source,
                          'target': target}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples
