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

"""GLUE finetuning/evaluation."""

from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import get_tokenizer
from megatron.model import ModelType
#from tasks.eval_utils import accuracy_func_provider
#from megatron.schedules import get_forward_backward_func
#from megatron.utils import get_ltor_masks_and_position_ids
from dataset import get_processed_dataset
# from dpr_encoder import DPRRetriever

def train(Dataset, FtDataset, model_provider,
                             forward_step, model_type, finetune):

    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()
        tokenizer = get_tokenizer()

        task_dataset = Dataset
        task_train_dataset = task_dataset["train"]
        task_valid_dataset = task_dataset["valid"]

        print(len(task_valid_dataset))
        print(len(task_train_dataset))

        train_dataset = FtDataset(args.task, task_train_dataset, 
                                args.max_seq_length, args.max_seq_length_dec) 
        valid_dataset = FtDataset(args.task, task_valid_dataset, 
                                args.max_seq_length, args.max_seq_length_dec)

        return train_dataset, valid_dataset


    """Finetune/evaluate."""
    finetune(train_valid_datasets_provider, model_provider, 
            model_type=model_type, forward_step=forward_step)

def main():
    args = get_args()
    from tasks.finetune_utils import finetune

    model_type = ModelType.encoder_or_decoder
    from dataset import RetroFtDataset as FtDataset
    from gpt_forward import forward_step
    from gpt_forward import model_provider
    args.max_seq_length = args.seq_length
    args.max_seq_length_dec = 0

    # Dataset = get_dataset_class(args.task)
    Dataset = get_processed_dataset(args.task, args.data_folder)
   
    # use valid_data to set evaluation dataset
    args.valid_data = [args.task]

    # set orig_micro_batch_size to be compatible with eval_utils
    if args.epochs <= 0:
        args.orig_micro_batch_size = args.micro_batch_size

    # if args.task.lower() == 'eli5':
    #   dpr_encoder = DPRRetriever(args.dpr_mode, args.faiss_ckpt, args.original_db_file)
    # else:
    #   dpr_encoder = None

    train(Dataset, FtDataset, model_provider, \
            forward_step, model_type, finetune)
