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

from functools import partial
import h5py
import json
import numpy as np
import os
import time
import torch

from megatron import (
    get_args,
    mpu,
    print_rank_0,
)
from megatron.data.data_samplers import MegatronPretrainingSampler
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.model import ModelType
from megatron.schedules import get_forward_backward_func
from megatron.training import (
    setup_model_and_optimizer,
)
from pretrain_bert import (
    # forward_step as forward_step_func,
    get_batch,
    model_provider,
    # train_valid_test_datasets_provider,
)

from ..preprocess.utils import get_sampled_chunk_index_path
from .chunk_dataset import BertChunkDataset

# >>>
from lutil import pax, print_seq
# <<<


def loss_func(loss_mask, sentence_order, output_tensor, non_loss_data):
    assert non_loss_data
    return output_tensor


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()

    # Get the batch.
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)

    # >>>
    # pax(0, {
    #     "data_iterator" : data_iterator,
    #     "tokens" : tokens,
    #     "types" : types,
    #     "sentence_order" : sentence_order,
    #     "loss_mask" : loss_mask,
    #     "lm_labels" : lm_labels,
    #     "padding_mask" : padding_mask,
    # })
    # <<<

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)

    # >>>
    # print(tokens)
    # pax(0, {
    #     "model" : model,
    #     "tokens" : tokens,
    #     "output_tensor" : output_tensor,
    # })
    # <<<

    return output_tensor, partial(loss_func, loss_mask, sentence_order)

def embed_batches(args, models, data_loader):

    # Data iterator.
    data_iterator = iter(data_loader)

    # Eval mode.
    for m in models:
        m.eval()

    # Compute embeddings.
    forward_backward_func = get_forward_backward_func()
    with torch.no_grad():

        # Iterate batches.
        n_batches = int(np.ceil(len(data_iterator) / args.micro_batch_size))
        dataset_start_time = time.time()
        batch_times = []
        for batch_index in range(n_batches):

            # Forward pass.
            batch_start_time = time.time()
            output_tensors = forward_backward_func(
                forward_step,
                data_iterator,
                models,
                optimizer = None,
                timers = None,
                forward_only = True,
                collect_non_loss_data = True,
            )
            batch_end_time = time.time()
            batch_times.append(batch_end_time - batch_start_time)
            mean_batch_time = sum(batch_times[-8:]) / min(len(batch_times), 8)

            # Progress.
            est_dataset_time = (batch_end_time - dataset_start_time) + \
                (n_batches - batch_index - 1) * mean_batch_time
            samples_per_sec = len(data_loader.dataset) / est_dataset_time
            print_rank_0("batch %d / %d [%d] ... %.3f samples/sec [ 47b = %.1f node days ]." % (
                batch_index,
                n_batches,
                data_loader.dataset.batch_chunk_lens[batch_index],
                samples_per_sec,
                (47e9 / samples_per_sec) / 16 / (24 * 3600),
            ))


def get_chunk_data_loader(args, data_metas, timer):

    # Token datasets.
    indexed_datasets = \
        [ make_indexed_dataset(m["prefix"], "mmap", True) for m in data_metas ]

    # Chunk index.
    chunk_index_path = get_sampled_chunk_index_path(args.retrieval_workdir)
    f = h5py.File(chunk_index_path, "r")
    dataset_offsets = np.copy(f["dataset_offsets_valid"])
    chunk_index = np.copy(f["chunks_valid"])
    f.close()

    # Chunk dataset.
    dataset = BertChunkDataset(
        indexed_datasets,
        dataset_offsets,
        chunk_index,
        args.retrieval_chunk_len,
        args.seq_length,
        args.micro_batch_size,

        masked_lm_prob=args.mask_prob,
        seed=args.seed,

        # >>>
        # binary_head = args.bert_binary_head,
        binary_head = False, # allows len(segments) == 1
        # <<<
    )

    # Megatron sampler.
    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(chunk_index),
        consumed_samples=0,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size())

    # Torch data loader.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    # >>>
    # pax({
    #     # "chunk_index" : chunk_index,
    #     "dataset" : dataset,
    #     "batch_sampler" : batch_sampler,
    #     "data_loader" : data_loader,
    # })
    # <<<

    return data_loader

def embed_chunks(args, timer):

    # # Embedding workdir.
    # workdir = os.path.join(args.retrieval_workdir, "embed")
    # os.makedirs(workdir, exist_ok = True)

    with open(os.path.join(args.retrieval_workdir, "order.json")) as f:
        data_metas = json.load(f)

    data_loader = get_chunk_data_loader(args, data_metas, timer)
    # data_iterator = iter(data_loader)

    # pax({
    #     "data_metas" : data_metas,
    #     "data_loader" : data_loader,
    # })

    models, optimizer, opt_param_scheduler = \
        setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

    # embed_batches(args, models, data_iterator)
    embed_batches(args, models, data_loader)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# eof
