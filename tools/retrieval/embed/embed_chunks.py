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
# from megatron.data.data_samplers import MegatronPretrainingSampler
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


def get_n_chunks(args):
    chunk_index_path = get_sampled_chunk_index_path(args.retrieval_workdir)
    f = h5py.File(chunk_index_path, "r")
    n_chunks = len(f["chunks_valid"])
    f.close()
    return n_chunks


# def get_missing_embedding_blocks(args, workdir, data_loader):
def get_missing_embedding_blocks(args, workdir, n_chunks):

    # n_chunks = len(data_loader)

    block_size = args.retrieval_block_size
    block_start_idxs = list(range(0, n_chunks, block_size))
    block_end_idxs = [ min(n_chunks, i + block_size) for i in block_start_idxs ]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # block_path_map = {r:os.path.join(workdir, "%d-%d.hdf5" % r)
    #                   for r in block_ranges}
    # missing_block_path_map = {r:p
    #                           for r, p in block_path_map.items()
    #                           if not os.path.exists(p)}
    all_block_items = [{
        "range" : r,
        "path" : os.path.join(workdir, "%d-%d.hdf5" % r)
    } for r in block_ranges]
    missing_block_items = [item
                           for item in all_block_items
                           if not os.path.exists(item["path"])]


    data_parallel_rank = mpu.get_data_parallel_rank()
    data_parallel_world_size = mpu.get_data_parallel_world_size()
    rank_missing_block_items = missing_block_items[data_parallel_rank:len(missing_block_items):data_parallel_world_size]

    # >>>
    # print_seq("my inital ranges are %s." % ", ".join(str(i["range"]) for i in rank_missing_block_items[:3]))
    # pax(0, {
    #     "all_block_items / len" : len(all_block_items),
    #     "missing_block_items / len" : len(missing_block_items),
    #     # "missing_block_items / 0" : missing_block_items[0],
    #     "rank_missing_block_items / len" : len(rank_missing_block_items),
    #     "workdir" : workdir,
    #     # "data_loader / len" : len(data_loader),
    #     "n_chunks" : n_chunks,
    #     "block_size" : block_size,
    # })
    # <<<

    return rank_missing_block_items

    
# def get_chunk_data_loader(args, data_metas, timer):

#     # Token datasets.
#     indexed_datasets = \
#         [ make_indexed_dataset(m["prefix"], "mmap", True) for m in data_metas ]

#     # Chunk index.
#     chunk_index_path = get_sampled_chunk_index_path(args.retrieval_workdir)
#     f = h5py.File(chunk_index_path, "r")
#     dataset_offsets = np.copy(f["dataset_offsets_valid"])
#     chunk_index = np.copy(f["chunks_valid"])
#     f.close()

#     # Chunk dataset.
#     dataset = BertChunkDataset(
#         indexed_datasets,
#         dataset_offsets,
#         chunk_index,
#         args.retrieval_chunk_len,
#         args.seq_length,
#         args.micro_batch_size,

#         masked_lm_prob=args.mask_prob,
#         seed=args.seed,

#         # >>>
#         # binary_head = args.bert_binary_head,
#         binary_head = False, # allows len(segments) == 1
#         # <<<
#     )

#     # Megatron sampler.
#     batch_sampler = MegatronPretrainingSampler(
#         total_samples=len(chunk_index),
#         consumed_samples=0,
#         micro_batch_size=args.micro_batch_size,
#         data_parallel_rank=mpu.get_data_parallel_rank(),
#         data_parallel_size=mpu.get_data_parallel_world_size())

#     # Torch data loader.
#     data_loader = torch.utils.data.DataLoader(dataset,
#                                               batch_sampler=batch_sampler,
#                                               num_workers=args.num_workers,
#                                               pin_memory=True)

#     # >>>
#     # pax({
#     #     # "chunk_index" : chunk_index,
#     #     "dataset" : dataset,
#     #     "batch_sampler" : batch_sampler,
#     #     "data_loader" : data_loader,
#     # })
#     # <<<

#     return data_loader
def get_shared_dataset_info(args):

    # Load dataset metadata.
    with open(os.path.join(args.retrieval_workdir, "order.json")) as f:
        data_metas = json.load(f)

    # Token datasets.
    indexed_datasets = \
        [ make_indexed_dataset(m["prefix"], "mmap", True) for m in data_metas ]

    # Chunk index.
    chunk_index_path = get_sampled_chunk_index_path(args.retrieval_workdir)
    f = h5py.File(chunk_index_path, "r")
    dataset_offsets = np.copy(f["dataset_offsets_valid"])
    chunk_index = np.copy(f["chunks_valid"])
    f.close()

    dataset_ids = []
    for i in range(len(dataset_offsets) - 1):
        dataset_ids.append([i] * (dataset_offsets[i+1] - dataset_offsets[i]))
    dataset_ids = [ i for ii in dataset_ids for i in ii ]

    return {
        "data_metas" : data_metas,
        "indexed_datasets" : indexed_datasets,
        # "dataset_offsets" : dataset_offsets,
        "dataset_ids" : dataset_ids,
        "chunk_index" : chunk_index,
    }

def get_block_data_loader(args, shared_dataset_info, chunk_start_id, chunk_end_id):

    # pax(0, {"shared_dataset_info": shared_dataset_info})

    # Chunk dataset.
    # t = time.time()
    dataset = BertChunkDataset(
        indexed_datasets = shared_dataset_info["indexed_datasets"],
        dataset_ids = shared_dataset_info["dataset_ids"][chunk_start_id:chunk_end_id],
        chunk_index = shared_dataset_info["chunk_index"][chunk_start_id:chunk_end_id],
        # chunk_start_idx = chunk_start_idx,
        # chunk_end_idx = chunk_end_idx,
        max_chunk_len = args.retrieval_chunk_len,
        max_seq_len = args.seq_length,
        micro_batch_size = args.micro_batch_size,
        
        masked_lm_prob = args.mask_prob,
        seed = args.seed,
        
        # >>>
        # binary_head = args.bert_binary_head,
        binary_head = False, # allows len(segments) == 1
        # <<<
    )

    # t = time.time() - t
    # print_seq("chunk dataset, %.3f sec." % t)

    # Megatron sampler.
    # >>>
    # batch_sampler = MegatronPretrainingSampler(
    #     # total_samples = len(chunk_index),
    #     total_samples = chunk_end_id - chunk_start_id,
    #     consumed_samples = 0,
    #     micro_batch_size = args.micro_batch_size,
    #     data_parallel_rank = mpu.get_data_parallel_rank(),
    #     data_parallel_size = mpu.get_data_parallel_world_size())
    # +++
    from torch.utils.data import BatchSampler, SequentialSampler
    batch_sampler = BatchSampler(
        sampler = SequentialSampler(dataset),
        batch_size = args.micro_batch_size,
        drop_last = False,
    )
    # pax(0, {"batch_sampler": batch_sampler})
    # +++

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


# def embed_batches(args, models, data_loader, missing_embedding_blocks):
def embed_batches(args, models, data_loader,
                  # get_more_data_loaders,
):

    # Data iterator.
    data_iterator = iter(data_loader)

    # Eval mode.
    for m in models:
        m.eval()

    # >>>
    # get_more_data_loaders() # good.
    # <<<

    # Compute embeddings.
    forward_backward_func = get_forward_backward_func()
    with torch.no_grad():

        # Iterate batches.
        # n_batches = int(np.ceil(len(data_iterator) / args.micro_batch_size))
        n_batches = len(data_iterator)
        # pax(0, {
        #     "data_loader / len" : len(data_loader),
        #     "data_iterator / len" : len(data_iterator),
        #     "n_batches" : n_batches,
        # })
        dataset_start_time = time.time()
        batch_times = []
        embeddings = []
        # >>>
        # get_more_data_loaders() # good.
        # <<<
        for batch_index in range(n_batches):

            # Forward pass.
            batch_start_time = time.time()
            # >>>
            # get_more_data_loaders()
            # <<<
            output_tensors = forward_backward_func(
                forward_step,
                data_iterator,
                models,
                optimizer = None,
                timers = None,
                forward_only = True,
                collect_non_loss_data = True,
            )
            # >>>
            # get_more_data_loaders()
            # <<<
            assert len(output_tensors) == 1, "assert len(models) == 1 before this"
            embeddings.append(output_tensors[0].cpu().numpy())
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

    return np.concatenate(embeddings, axis = 0)


def embed_blocks(args, models, shared_data_info, missing_embedding_blocks):

    for block_index, block_info in enumerate(missing_embedding_blocks):

        print_rank_0("embed block %d / %d ... %s." % (
            block_index,
            len(missing_embedding_blocks),
            os.path.basename(block_info["path"]),
        ))

        # raise Exception("hi.")

        # Data loader.
        def get_data_loader():
            return get_block_data_loader(args,
                                         shared_data_info,
                                         *block_info["range"])
        def get_more_data_loaders():
            get_data_loader()
            get_data_loader()
            raise Exception("hi.")

        data_loader = get_data_loader()

        embeddings = embed_batches(args, models, data_loader,
                                   # get_more_data_loaders,
        )

        # >>>
        # get_more_data_loaders()
        # <<<

        f = h5py.File(block_info["path"], "w")
        f.create_dataset("data", data = embeddings)
        f.close()

        # pax(0, {
        #     "embeddings" : embeddings,
        #     "data_loader" : data_loader,
        #     "data_loader / len" : len(data_loader),
        #     "block_info" : block_info,
        # })

    # >>>
    torch.distributed.barrier()
    print_seq("i am done.")
    # <<<


def embed_chunks(args, timer):

    # >>>
    # from .test_huggingface import test_huggingface
    # test_huggingface(args, timer)
    # raise Exception("hi.")
    # <<<

    # print_seq("i am data rank %d." % mpu.get_data_parallel_rank())

    # Embedding workdir.
    workdir = os.path.join(args.retrieval_workdir, "embed")
    os.makedirs(workdir, exist_ok = True)

    # Load model.
    models, optimizer, opt_param_scheduler = \
        setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

    # Share dataset info (indexed datasets, chunk index, etc.).
    # t = time.time()
    shared_dataset_info = get_shared_dataset_info(args)
    # t = time.time() - t
    # print_seq("shared dataset info, time = %.3f." % t)

    # >>>
    # bert_chunk_lens = [ a[3] for a in shared_dataset_info["chunk_index"] ]
    # pax(0, {
    #     "shared_dataset_info" : shared_dataset_info,
    #     "bert_chunks_lens" : bert_chunk_lens[:20],
    #     "bert_chunks_lens / len" : len(bert_chunk_lens),
    #     "mean" : np.mean(bert_chunk_lens),
    #     "var" : np.var(bert_chunk_lens),
    # })
    # <<<

    # Missing embedding blocks (stored on disk).
    n_chunks = get_n_chunks(args)
    missing_embedding_blocks = \
        get_missing_embedding_blocks(args, workdir, n_chunks) # data_loader)

    # Prevent missing file race condition.
    torch.distributed.barrier()

    # pax(0, {
    #     "missing_embedding_blocks" : missing_embedding_blocks,
    #     "n_chunks" : n_chunks,
    # })

    # # >>>
    # # Data loader.
    # # t = time.time()
    # data_loader = get_chunk_data_loader(args, data_metas, timer)
    # # t = time.time() - t
    # # print_seq("data_loader, %.3f sec." % t)
    # # <<<

    # pax(0, {
    #     "data_metas" : data_metas,
    #     # "data_loader" : data_loader,
    #     "data_loader / len" : len(data_loader),
    # })

    # Embed batches.
    # embed_batches(args, models, data_loader, missing_embedding_blocks)
    embed_blocks(args, models, shared_dataset_info, missing_embedding_blocks)

    raise Exception("unsort chunks.")

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# eof
