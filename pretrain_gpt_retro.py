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

"""Pretrain GPT w/ Retro"""

from functools import partial
import torch

from megatron import get_args, get_retro_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron import print_rank_0
# from megatron.data.blendable_dataset import BlendableDataset
# from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
# from megatron.utils import average_losses_across_data_parallel_group
from tools.retro.pretraining.retro_dataset import get_retro_datasets

from pretrain_gpt import (
    loss_func,
    model_provider,
    train_valid_test_datasets_provider as standard_datasets_provider,
)

# >>>
from lutil import pax
# <<<


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


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    retro_args = get_retro_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    if args.retro_add_retriever:
        keys += 'neighbor_tokens',

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = mpu.broadcast_data(keys, data, datatype)

    # pax(0, {"data_b": data_b})

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    if args.retro_add_retriever:
        # note: [bs * l * k, r]
        # note: 2x == neighbor, continuation
        neighbor_tokens = data_b['neighbor_tokens'] \
            .view(-1, retro_args.retro_gpt_retrieved_length).long()
        # pax(0, {"neighbor_tokens": neighbor_tokens})

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    if args.retro_add_retriever:
        _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
            neighbor_tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        neighbor_attention_mask = None
        return tokens, labels, loss_mask, attention_mask, position_ids, \
               neighbor_tokens, neighbor_attention_mask, neighbor_position_ids
    else:
        return tokens, labels, loss_mask, attention_mask, position_ids


# def loss_func(loss_mask, output_tensor):
#     losses = output_tensor.float()
#     loss_mask = loss_mask.view(-1).float()
#     loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

#     # Reduce loss for logging.
#     averaged_loss = average_losses_across_data_parallel_group([loss])

#     return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    if args.retro_add_retriever:
        tokens, labels, loss_mask, attention_mask, position_ids, \
            neighbor_tokens, neighbor_attention_mask, neighbor_position_ids \
            = get_batch(data_iterator)
        # >>>
        # pax(0, {"tokens": tokens, "neighbor_tokens": neighbor_tokens})
        # <<<
    else:
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
        neighbor_tokens, neighbor_attention_mask, neighbor_position_ids = None, None, None
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          ret_int_ids=neighbor_tokens,
                          ret_position_ids=neighbor_position_ids,
                          ret_attn_mask=neighbor_attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


# def train_valid_test_datasets_provider(train_val_test_num_samples):
#     """Build train, valid, and test datasets."""
#     args = get_args()

#     print_rank_0('> building train, validation, and test datasets '
#                  'for GPT ...')
#     train_ds1, valid_ds, test_ds = build_train_valid_test_datasets(
#         data_prefix=args.data_path,
#         data_impl=args.data_impl,
#         splits_string=args.split,
#         train_valid_test_num_samples=train_val_test_num_samples,
#         seq_length=args.seq_length,
#         seed=args.seed,
#         skip_warmup=(not args.mmap_warmup))
#     print_rank_0("> finished creating finetuning GPT datasets ...")

#     # train_ds = train_ds1
#     pax(0, {"train_val_test_num_samples": train_val_test_num_samples})
#     # >>>
#     train_ds = BlendableDataset([train_ds1], [args.weight])
#     valid_ds = BlendableDataset([valid_ds], [args.weight])
#     if args.neighbors_path:
#         import h5py
#         train_ds.neighbors = h5py.File(args.neighbors_path, 'r')
#         train_ds.database = h5py.File(args.database_path, 'r')

#         valid_ds.neighbors = h5py.File(args.valid_neighbors_path, 'r')
#         valid_ds.database = h5py.File(args.valid_database_path, 'r')
#     # +++
#     raise Exception("moved to new method.")
#     train_ds, valid_ds, test_ds = \
#         [DummyRetroDataset.from_n_samples(n) for n in train_val_test_num_samples]
#     # <<<

#     return train_ds, valid_ds, test_ds
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    if args.retro_add_retriever:
        return get_retro_datasets()
    else:
        return standard_datasets_provider(train_val_test_num_samples)
# import numpy as np

# class DummyRetroDataset(torch.utils.data.Dataset):

#     def __init__(self, n_samples):

#         raise Exception("use retro dataset.")

#         super().__init__()

#         args = get_args()

#         self.n_samples = n_samples
#         self.seq_length = args.seq_length
#         self.chunk_length = args.retro_chunk_length
#         self.retrieved_length = args.retro_retrieved_length
#         self.nnbrs = args.retro_nnbrs
#         self.tokenizer = get_tokenizer()

#         self.n_chunks_per_seq = int(np.ceil(self.seq_length / self.chunk_length))

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, sample_idx):

#         token_ids = np.random.randint(0, 10000, self.seq_length)
#         nbr_token_ids = np.random.randint(0, 10000, (
#             self.n_chunks_per_seq, self.nnbrs, self.retrieved_length))

#         sample = {
#             "text" : token_ids,
#             "neighbor_tokens" : nbr_token_ids,
#         }

#         # pax(0, {
#         #     "token_ids" : "%d / %s" % (len(token_ids), str(token_ids)),
#         #     "nbr_token_ids" : nbr_token_ids,
#         #     "nbr_token_ids / size" : nbr_token_ids.size,
#         #     "sample" : sample,
#         # })

#         return sample

# def train_valid_test_datasets_provider(train_val_test_num_samples):
#     return [DummyRetroDataset(n) if n is not None else None
#             for n in train_val_test_num_samples]

# def add_validation_args(parser):
# def add_retro_args(parser):
#     """Text generation arguments."""
#     group = parser.add_argument_group(title='validation set')
#     # group.add_argument('--data-path2', nargs='*', default=None,
#     #                    help='Path to the training dataset. Accepted format:'
#     #                    '1) a single data path, 2) multiple datasets in the'
#     #                    'form: dataset1-weight dataset1-path dataset2-weight '
#     #                    'dataset2-path ...')
#     # group.add_argument('--weight', type=float, default=1)
#     # group.add_argument('--adaptor', action='store_true', default=False)
#     # group.add_argument('--project-size', type=int, default=256)
#     group.add_argument('--retro-cyclic-train-iters', type=int, default=None)
#     # group.add_argument('--stored-params', type=dict, default=dict())
#     group.add_argument('--retro-eval-ppl', action='store_true', default=False)
#     group.add_argument('--retro-debug', action='store_true', default=False)
#     group.add_argument('--retro-add-retriever', action='store_true', default=False)
#     # group.add_argument('--return-doc-ids', action='store_true', default=False)
#     # group.add_argument('--return-neighbor-ids', action='store_true', default=False)
#     # group.add_argument('--add-offset-doc-ids', action='store_true', default=False)
#     # group.add_argument('--offset-dict-path', type=str, default='')
#     # group.add_argument('--neighbors-path', type=str, default='')
#     # group.add_argument('--valid-neighbors-path', type=str, default='')
#     # group.add_argument('--database-path', type=str, default='')
#     # group.add_argument('--valid-database-path', type=str, default='')
#     group.add_argument('--retro-encoder-layers', type=int, default=2) # 12)
#     group.add_argument('--retro-encoder-hidden-dropout', type=float, default=0.1)
#     group.add_argument('--retro-encoder-attention-dropout', type=float, default=0.1)
#     # group.add_argument('--k', type=int, default=2)
#     # group.add_argument('--r', type=int, default=128)
#     # group.add_argument('--m', type=int, default=64)
#     # group.add_argument('--l', type=int, default=32)

#     group.add_argument("--retro-chunk-length", type=int, default=64)
#     group.add_argument("--retro-retrieved-length", type=int, default=128)
#     group.add_argument("--retro-nnbrs", type=int, default=2) # -pretraining

#     return parser


if __name__ == "__main__":

    # pax(0, {"hi.": "there."})

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             # extra_args_provider=add_retro_args)
    )
