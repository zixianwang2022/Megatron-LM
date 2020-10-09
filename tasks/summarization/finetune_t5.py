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

"""CNNDM finetuning/evaluation."""

from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.model.t5_model import T5Model
from tasks.eval_utils_t5 import accuracy_func_provider
from tasks.finetune_utils_t5 import finetune


def summarization(Dataset, name_from_datapath_func):

    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()
        tokenizer = get_tokenizer()

        train_dataset = Dataset('training', args.train_data,
                                tokenizer, args.seq_length,
                                args.decoder_seq_length)
        valid_dataset = Dataset('validation', args.valid_data,
                                tokenizer, args.seq_length,
                                args.decoder_seq_length)
        return train_dataset, valid_dataset

    def model_provider():
        """Build the model."""
        args = get_args()

        print_rank_0('building T5 model for {} ...'.format(args.task))

        return T5Model(num_tokentypes=2,
                       parallel_output=False)

    def single_dataset_provider(datapath):
        args = get_args()
        tokenizer = get_tokenizer()

        name = name_from_datapath_func(datapath)
        return Dataset(name, datapath, tokenizer,
                       args.seq_length, args.decoder_seq_length)

    def distributed_metrics_func_provider(datapath):
        """Provide metrics callback function."""
        return accuracy_func_provider(single_dataset_provider, datapath)

    def rank0_metrics_func_provider(datapath):
        """Provide metrics callback function."""
        return accuracy_func_provider(single_dataset_provider, datapath,
                                      rank0sampler=True)

    """Finetune/evaluate."""
    finetune(train_valid_datasets_provider,
             model_provider,
             end_of_epoch_callback_provider=distributed_metrics_func_provider,
             end_of_training_callback_provider=rank0_metrics_func_provider)


def main():
    args = get_args()

    if args.task == 'CNNDM':
        from tasks.summarization.cnndm import CNNDMDataset as Dataset

        def name_from_datapath(datapath):
            return datapath[0].split('/')[-1].split('.')[0]
    else:
        raise NotImplementedError('Summarization task {} is not implemented.'.format(
            args.task))

    summarization(Dataset, name_from_datapath)
