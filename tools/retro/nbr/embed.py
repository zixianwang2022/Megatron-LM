# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# from functools import partial
# import h5py
# import multiprocessing
# import numpy as np
# import os
# import time
# import torch
# from tqdm import tqdm

# from megatron import get_args, print_rank_0
# from megatron.data.gpt_dataset import build_train_valid_test_datasets
# from megatron.tokenizer.tokenizer import _GPT2BPETokenizer
# from megatron.training import (
#     build_train_valid_test_data_loaders,
#     update_train_iters,
# )
from tools.retro.embed import embed_text_datasets

from .dataset import get_dataset_map

# >>>
from lutil import pax
# <<<


# def add_validation_args(parser):
#     """Text generation arguments."""
#     group = parser.add_argument_group(title='validation set')
#     group.add_argument('--data-path2', nargs='*', default=None,
#                        help='Path to the training dataset. Accepted format:'
#                        '1) a single data path, 2) multiple datasets in the'
#                        'form: dataset1-weight dataset1-path dataset2-weight '
#                        'dataset2-path ...')
#     group.add_argument('--weight', type=float, default=0.5)
#     group.add_argument('--adaptor', action='store_true', default=False)
#     group.add_argument('--return_doc_ids', action='store_true', default=False)
#     group.add_argument('--return_neighbor_ids', action='store_true', default=False)
#     group.add_argument('--add_offset_doc_ids', action='store_true', default=False)
#     group.add_argument('--offset_dict_path', type=str, default='')
#     group.add_argument('--project-size', type=int, default=256)
#     group.add_argument('--stored_params', type=dict, default=dict())
#     group.add_argument('--eval_ppl', action='store_true', default=False)
#     parser.add_argument('--workers', type=int, default=100,
#                         help='Number of worker processes to launch')
#     parser.add_argument('--start', type=int, default=0,
#                         help='iteration start')
#     parser.add_argument('--end', type=int, default=0,
#                         help='iteration end')
#     group.add_argument('--neighbors_path', type=str, default='')
#     return parser


# def model_provider(pre_process=True, post_process=True):
#     """Build the model."""

#     print_rank_0('building GPT model ...')
#     model = GPTModel(
#         num_tokentypes=0,
#         parallel_output=True,
#         pre_process=pre_process,
#         post_process=post_process
#     )
#     return model


# def get_batch(data_iterator):
#     """Generate a batch"""
#     args = get_args()
#     tokenizer = get_tokenizer()

#     # Items and their type.
#     keys = ['text']
#     datatype = torch.int64

#     # Broadcast data.
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     data_b = mpu.broadcast_data(keys, data, datatype)

#     # Unpack.
#     tokens_ = data_b['text'].long()
#     labels = tokens_[:, 1:].contiguous()
#     tokens = tokens_[:, :-1].contiguous()

#     # Get the masks and postition ids.
#     attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
#         tokens,
#         tokenizer.eod,
#         args.reset_position_ids,
#         args.reset_attention_mask,
#         args.eod_mask_loss)

#     return tokens, labels, loss_mask, attention_mask, position_ids

# def get_batch_for_preprocess(data_iterator):
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     pax(0, {"data": data})
#     tokens_ = data['text']
#     tokens = tokens_[:, :-1].contiguous()
#     return tokens, [doc.item() for doc in data["doc_ids"]], data['idx']

# def get_batch_for_preprocess_by_data(data):
#     tokens_ = data['text']
#     tokens = tokens_[:, :-1].contiguous()
#     return tokens, [doc.item() for doc in data["doc_ids"]]



# def loss_func(loss_mask, output_tensor):
#     losses = output_tensor.float()
#     loss_mask = loss_mask.view(-1).float()
#     loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

#     # Reduce loss for logging.
#     averaged_loss = average_losses_across_data_parallel_group([loss])

#     return loss, {'lm loss': averaged_loss[0]}


# def forward_step(data_iterator, model):
#     """Forward step."""
#     args = get_args()
#     timers = get_timers()

#     # Get the batch.
#     timers('batch-generator').start()
#     tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
#         data_iterator)
#     timers('batch-generator').stop()

#     output_tensor = model(tokens, position_ids, attention_mask,
#                           labels=labels)

#     return output_tensor, partial(loss_func, loss_mask)


# def embed_pretraining_tokens(args, timer):
def embed_pretraining_chunks(args, workdir, timer):

    # Data stuff.
    dataset_map = get_dataset_map(args, workdir)

    # pax(0, {"dataset_map": dataset_map})

    # for key, dataset in dataset_map.items():
    #     embed_workdir = os.path.join(workdir, "embed", key)
    #     os.makedirs(embed_workdir, exist_ok = True)
    #     embed_dataset(args, embed_workdir, key, dataset)
    embed_text_datasets(args, dataset_map)

    raise Exception("finished embedding.")
