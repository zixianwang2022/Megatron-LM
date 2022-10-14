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
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from megatron import (
    get_args,
    get_tokenizer,
    mpu,
    print_rank_0,
)
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.model import BertModel, ModelType
from megatron.schedules import get_forward_backward_func
from megatron.training import (
    setup_model_and_optimizer,
)

from ..chunks.utils import get_chunk_index_path_map
from .chunk_dataset import BertEmbeddingDataset # BertChunkDataset
from .long_bert_chunks import print_longest_bert_chunks

# >>>
from lutil import pax, print_seq
# <<<


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0(" > build Bert model.")

    args = get_args()
    num_tokentypes = 2 if args.bert_binary_head else 0
    model = BertModel(
        num_tokentypes=num_tokentypes,
        add_binary_head=args.bert_binary_head,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process)

    return model


def get_dataset_map(args):

    # Load dataset metadata.
    with open(os.path.join(args.retro_workdir, "order.json")) as f:
        data_metas = json.load(f)

    # Token datasets.
    indexed_datasets = \
        [ make_indexed_dataset(m["prefix"], "mmap", True) for m in data_metas ]

    # Chunk index.
    chunk_index_path_map = get_chunk_index_path_map(args.retro_workdir)
    dataset_map = {}
    for key, chunk_index_path in chunk_index_path_map.items():

        # Load chunk index.
        f = h5py.File(chunk_index_path, "r")
        dataset_offsets = np.copy(f["dataset_offsets_valid"])
        chunk_index = np.copy(f["chunks_valid"])
        f.close()

        # Dataset ids.
        dataset_ids = []
        for i in range(len(dataset_offsets) - 1):
            dataset_ids.append([i] * (dataset_offsets[i+1] - dataset_offsets[i]))
        dataset_ids = [ i for ii in dataset_ids for i in ii ]

        # Dataset.
        dataset = BertChunkDataset(
            indexed_datasets = indexed_datasets,
            dataset_ids = dataset_ids,
            chunk_index = chunk_index,
            max_chunk_length = args.retro_chunk_length,
            max_model_seq_length = args.seq_length,
            masked_lm_prob = args.mask_prob,
            seed = args.seed,
        )

        dataset_map[key] = dataset

    return dataset_map


def get_missing_embedding_blocks(args, workdir, dataset):

    n_chunks = len(dataset)

    # Block ranges.
    block_size = args.retro_block_size
    block_start_idxs = list(range(0, n_chunks, block_size))
    block_end_idxs = [ min(n_chunks, i + block_size) for i in block_start_idxs ]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # All block files (existing + missing).
    n_digits = int(np.ceil(np.log(n_chunks) / np.log(10)) + 1)
    all_block_items = [{
        "range" : r,
        "path" : os.path.join(
            workdir,
            "%s-%s.hdf5" % tuple([ str(i).zfill(n_digits) for i in r ]),
        )
    } for r in block_ranges]

    # Delete corrupt files.
    if torch.distributed.get_rank() == 0:
        existing_block_paths = [item["path"]
                                for item in all_block_items
                                if os.path.exists(item["path"])]
        pbar = tqdm(existing_block_paths)
        for index, path in enumerate(pbar):
            pbar.set_description("verifying embedding block.")
            f = h5py.File(path, "r")
            try:
                assert f["data"].shape[1] == 1024
            except:
                raise Exception("delete block file.")
            finally:
                f.close()

    # Wait for files to be deleted.
    torch.distributed.barrier()

    # Filter missing files.
    missing_block_items = [item
                           for item in all_block_items
                           if not os.path.exists(item["path"])]

    # This rank's missing files.
    data_parallel_rank = mpu.get_data_parallel_rank()
    data_parallel_world_size = mpu.get_data_parallel_world_size()
    rank_missing_block_items = missing_block_items[data_parallel_rank:len(missing_block_items):data_parallel_world_size]

    # Extend rank's missing items (with None) such that all ranks have equal
    # length lists. This allows for easier tracking of global progress.
    n_missing_tensor = torch.cuda.LongTensor([len(rank_missing_block_items)])
    torch.distributed.all_reduce(n_missing_tensor,
                                 op = torch.distributed.ReduceOp.MAX)
    max_n_missing = n_missing_tensor.item()
    rank_missing_block_items += \
        [None] * (max_n_missing - len(rank_missing_block_items))

    return rank_missing_block_items


def collate_batch(samples):
    """Collate samples of various lengths.

    This collate function handles samples with various sequence lengths, by
    padding 'text' arrays with pad_id, and other arrays with 0.
    """

    n_samples = len(samples)
    keys = list(samples[0].keys())
    tokenizer = get_tokenizer()

    # Max sample length across all samples.
    max_length_map = { key:0 for key in keys }
    for sample in samples:
        for key in keys:
            value_length = \
                len(sample[key]) if isinstance(sample[key], np.ndarray) else None
            max_length_map[key] = None \
                if value_length is None else \
                   max(max_length_map[key], value_length)

    # Pad samples.
    padded_samples = []
    for sample in samples:
        padded_sample = {}
        for key in keys:
            padded_sample[key] = \
                np.pad(
                    sample[key],
                    (0, max_length_map[key] - len(sample[key])),
                    mode = "constant",
                    constant_values = tokenizer.pad_id if key == "text" else 0,
                ) \
                if isinstance(sample[key], np.ndarray) else \
                   sample[key]
        padded_samples.append(padded_sample)

    # Build batch with padded samples.
    batch = default_collate(padded_samples)

    return batch


def get_block_data_loader(args, full_dataset, chunk_start_idx, chunk_end_idx):
    """Build data loader over data subset.

    Get a subset of the dataset (from start_idx -> end_idx), and wrap it in
    a sequential sampler and data loader.
    """

    # Dataset subset.
    block_dataset = Subset(full_dataset, range(chunk_start_idx, chunk_end_idx))

    # Sequential & batch samplers.
    batch_sampler = BatchSampler(
        sampler = SequentialSampler(block_dataset),
        batch_size = args.micro_batch_size,
        drop_last = False,
    )

    # Data loader.
    data_loader = DataLoader(block_dataset,
                             batch_sampler = batch_sampler,
                             num_workers = args.num_workers,
                             pin_memory = True,
                             collate_fn = collate_batch)

    return data_loader


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask',
            'seq_length']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()
    seq_lengths = data_b['seq_length'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask, \
        seq_lengths


def loss_func(loss_mask, sentence_order, seq_lengths,
              output_tensor, non_loss_data):
    """Loss function. Sequence lengths returned here for progress print-outs."""
    assert non_loss_data
    return seq_lengths, output_tensor


def forward_step(data_iterator, model):
    """Forward step."""

    args = get_args()

    # Get the batch.
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask, \
        seq_lengths = get_batch(data_iterator)

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask, sentence_order,
                                  seq_lengths)


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
        n_batches = len(data_iterator)
        dataset_start_time = time.time()
        batch_times = []
        embeddings = []
        for batch_index in range(n_batches):

            # Forward pass.
            batch_start_time = time.time()
            results = forward_backward_func(
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

            assert len(results) == 1, "assert len(models) == 1 before this"
            seq_lengths, output_tensor = results[0]
            embeddings.append(output_tensor.cpu().numpy())

            # Progress.
            if batch_index % 5 == 0:
                est_dataset_time = (batch_end_time - dataset_start_time) + \
                    (n_batches - batch_index - 1) * mean_batch_time
                samples_per_sec = len(data_loader.dataset) / est_dataset_time
                print_rank_0("batch %d / %d [%d] ... %.3f samples/sec [ 47b = %.1f node days ]." % (
                    batch_index,
                    n_batches,
                    seq_lengths.max().item(),
                    samples_per_sec,
                    (47e9 / samples_per_sec) / 16 / (24 * 3600),
                ))

    return np.concatenate(embeddings, axis = 0)


# def embed_blocks(args, models, prefix, dataset, missing_embedding_blocks):
def embed_blocks(args, models, workdir, dataset, missing_embedding_blocks):

    # Iterate blocks.
    for block_index, block_info in enumerate(missing_embedding_blocks):

        # Missing block lists are extended with None to have equal-length
        # lists. Skip the Nones.
        if block_info is not None:

            # print_rank_0("embed '%s' block %d / %d ... %s." % (
            #     prefix,
            #     block_index,
            #     len(missing_embedding_blocks),
            #     os.path.basename(block_info["path"]),
            # ))
            print_rank_0("embed block %d / %d ... %s." % (
                block_index,
                len(missing_embedding_blocks),
                workdir,
            ))

            # Data loader.
            data_loader = get_block_data_loader(args,dataset,*block_info["range"])

            # Embed block.
            embeddings = embed_batches(args, models, data_loader)

            # Save embeddings.
            f = h5py.File(block_info["path"], "w")
            f.create_dataset("data", data = embeddings)
            f.close()

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()


# def embed_dataset_chunks(args, workdir, models, prefix, dataset):

#     # Dataset workdir.
#     workdir = os.path.join(workdir, prefix)
#     os.makedirs(workdir, exist_ok = True)

#     # Missing embedding blocks (stored on disk).
#     missing_embedding_blocks = get_missing_embedding_blocks(args,workdir,dataset)

#     # Prevent missing file race condition.
#     torch.distributed.barrier()

#     # Embed batches.
#     embed_blocks(args, models, prefix, dataset, missing_embedding_blocks)


# def embed_chunks(args, timer):

#     # Embedding workdir.
#     workdir = os.path.join(args.retro_workdir, "embed")
#     os.makedirs(workdir, exist_ok = True)

#     # Load model.
#     models, optimizer, opt_param_scheduler = \
#         setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

#     # Dataset infos (indexed datasets, chunk index, etc.).
#     dataset_map = get_dataset_map(args)

#     # >>>
#     # del dataset_map["full"]
#     # <<<

#     # Embed each (i.e., full, sampled) dataset.
#     for prefix, dataset in dataset_map.items():
#         print_rank_0(" > embed '%s' chunks. [ count %d ]" %
#                      (prefix, len(dataset)))
#         embed_dataset_chunks(args, workdir, models, prefix, dataset)
def embed_text_dataset(args, models, workdir, text_dataset):

    # Dataset workdir.
    os.makedirs(workdir, exist_ok = True)

    # Bert embedding dataset.
    embedding_dataset = BertEmbeddingDataset(text_dataset)

    # Missing embedding blocks (stored on disk).
    missing_embedding_blocks = get_missing_embedding_blocks(args,
                                                            workdir,
                                                            embedding_dataset)

    # pax(0, {
    #     "missing_embedding_blocks" : missing_embedding_blocks,
    #     "workdir" : workdir,
    #     "text_dataset" : text_dataset,
    #     "embedding_dataset" : embedding_dataset,
    # })

    # Prevent missing file race condition.
    torch.distributed.barrier()

    # Embed batches.
    embed_blocks(args, models, workdir, embedding_dataset,
                 missing_embedding_blocks)


def embed_text_datasets(args, text_dataset_map):

    # Load model.
    models, optimizer, opt_param_scheduler = \
        setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

    # Embed each (i.e., full, sampled) dataset.
    for workdir, text_dataset in text_dataset_map.items():
        print_rank_0(" > embed %d seqs ... %s." %
                     (len(text_dataset), workdir))
        embed_text_dataset(args, models, workdir, text_dataset)


def embed_corpus_chunks(args, timer):

    raise Exception("call embed_text_datasets().")

    # Dataset infos (indexed datasets, chunk index, etc.).
    dataset_map = get_dataset_map(args)

    # embed_text_datasets
