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
import os
# import time
import torch
from tqdm import tqdm

from megatron import get_args, print_rank_0
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.tokenizer.tokenizer import _GPT2BPETokenizer
from megatron.training import (
    build_train_valid_test_data_loaders,
    update_train_iters,
)
from tools.retrieval.embed import embed_text_datasets

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


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.retro_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating pretrained GPT datasets ...")

    # train_ds = train_ds1
    # train_ds = BlendableDataset([train_ds1, train_ds2], [args.weight, 1 - args.weight])

    return train_ds, valid_ds, test_ds


# class TextChunkDataset(torch.utils.data.Dataset):
class SeqToChunkGPTDataset(torch.utils.data.Dataset):

    def __init__(self, args, seq_dataset):

        super().__init__()

        self.seq_dataset = seq_dataset

        self.seq_length = args.retro_seq_length
        self.chunk_length = args.retro_chunk_length
        assert self.seq_length % self.chunk_length == 0
        self.n_chunk_seq_ratio = int(self.seq_length / self.chunk_length)

        self.n_seqs = len(seq_dataset)
        self.n_chunks = self.n_seqs * self.n_chunk_seq_ratio

        # >>>
        self.gpt_tokenizer = _GPT2BPETokenizer(
            vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json",
            merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt",
        )
        # <<<

        # pax(0, {
        #     "seq_length" : self.seq_length,
        #     "chunk_length" : self.chunk_length,
        #     "n_chunk_seq_ratio" : self.n_chunk_seq_ratio,
        #     "n_seqs" : self.n_seqs,
        #     "n_chunks" : self.n_chunks,
        # })

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):

        seq_idx = idx // self.n_chunk_seq_ratio
        chunk_idx = idx % self.n_chunk_seq_ratio

        seq_token_ids = self.seq_dataset[seq_idx]["text"]
        # pax(0, {
        #     "seq_dataset" : self.seq_dataset,
        #     "seq_token_ids" : seq_token_ids,
        # })
        # assert len(seq_token_ids) == self.seq_length, \
        #     "len(seq_token_ids) == %d." % len(seq_token_ids)

        token_start_idx = chunk_idx * self.chunk_length
        token_end_idx = token_start_idx + self.chunk_length
        chunk_token_ids = seq_token_ids[token_start_idx:token_end_idx]
        chunk_text = self.gpt_tokenizer.detokenize(chunk_token_ids)

        # pax(0, {
        #     "seq_token_ids" : seq_token_ids,
        #     "chunk_token_ids" : chunk_token_ids,
        #     "chunk_text" : chunk_text,
        #     "seq_idx" : seq_idx,
        #     "chunk_idx" : chunk_idx,
        #     "token_start_idx" : token_start_idx,
        #     "token_end_idx" : token_end_idx,
        # })

        return chunk_text


# def embed_pretraining_tokens(args, timer):
def embed_pretraining_chunks(args, workdir, timer):

    # Update train iters.
    update_train_iters(args)

    # pax(0, {"args": args})

    # args.start = 0
    # args.end = args.train_samples
    # args.iteration = args.start
    # args.consumed_train_samples = args.start  # consumed samples == iterations (bs=1)
    args.iteration = 0
    args.consumed_train_samples = 0

    # Data stuff.
    print_rank_0(" > data loader.")
    train_data_loader, valid_data_loader, test_data_loader \
        = build_train_valid_test_data_loaders(
            train_valid_test_datasets_provider)
    data_loader_map = {
        "train" : train_data_loader,
        "valid" : valid_data_loader,
        "test" : test_data_loader,
    }
    dataset_map = {
        os.path.join(workdir, "embed", key) :
        SeqToChunkGPTDataset(args, loader.dataset)
        for key, loader in data_loader_map.items()
        if loader
    }

    # pax(0, {"dataset_map": dataset_map})

    # for key, dataset in dataset_map.items():
    #     embed_workdir = os.path.join(workdir, "embed", key)
    #     os.makedirs(embed_workdir, exist_ok = True)
    #     embed_dataset(args, embed_workdir, key, dataset)
    embed_text_datasets(args, dataset_map)

    raise Exception("finished embedding.")
