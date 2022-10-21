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
import numpy as np
import os
import time
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from megatron import get_args, get_tokenizer, mpu, print_rank_0
from megatron.model import BertModel, ModelType
from megatron.schedules import get_forward_backward_func
from megatron.training import setup_model_and_optimizer

from .dataset import BertEmbeddingDataset

# >>>
from lutil import pax, print_seq
from lutil.pax import print_mem_stats, get_mem_stats_str
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


# def get_missing_embedding_blocks(args, workdir, dataset):
def get_missing_embedding_blocks(workdir, dataset, block_size):

    n_samples = len(dataset)

    # Block ranges.
    # block_size = args.retro_block_size
    block_start_idxs = list(range(0, n_samples, block_size))
    block_end_idxs = [ min(n_samples, i + block_size) for i in block_start_idxs ]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # All block files (existing + missing).
    n_digits = int(np.ceil(np.log(n_samples) / np.log(10)) + 1)
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

    # >>>
    # print_seq("missing blocks [%d] : %s ... %s." % (
    #     len(rank_missing_block_items),
    #     str(rank_missing_block_items[0]["range"]),
    #     str(rank_missing_block_items[-1]["range"]) if rank_missing_block_items[-1] else str(rank_missing_block_items[-2]["range"]),
    # ))
    # <<<

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

    # >>>
    # if batch["text"].shape[1] > 400:
    #     pax(0, {"batch": batch})
    # <<<

    return batch


# def get_block_data_loader(args, full_dataset, sample_start_idx, sample_end_idx):
def get_block_data_loader(full_dataset, sample_start_idx, sample_end_idx):
    """Build data loader over data subset.

    Get a subset of the dataset (from start_idx -> end_idx), and wrap it in
    a sequential sampler and data loader.
    """

    args = get_args()

    # Dataset subset.
    block_dataset = Subset(full_dataset, range(sample_start_idx, sample_end_idx))

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
              # input_tensor, # ** temporary **
              output_tensor, non_loss_data):
    """Loss function. Sequence lengths returned here for progress print-outs."""
    assert non_loss_data
    return seq_lengths, output_tensor
    # return seq_lengths, input_tensor, output_tensor


def forward_step(data_iterator, model):
    """Forward step."""

    args = get_args()

    # Get the batch.
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask, \
        seq_lengths = get_batch(data_iterator)

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    # print_mem_stats("BEFORE")
    output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)
    # print_mem_stats("AFTER")

    return output_tensor, partial(loss_func, loss_mask, sentence_order,
                                  seq_lengths)


# def embed_batches(args, models, data_loader):
def embed_batches(models, data_loader):

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
            # >>>
            # print_mem_stats("batch %d / %d [ sq %d ]" % (
            #     batch_index,
            #     n_batches,
            #     results[0][0].max().item(),
            #     # results[0][1].shape[1],
            # ))
            # if batch_index == 30:
            #     exit(0)
            # continue
            # <<<
            batch_end_time = time.time()
            batch_times.append(batch_end_time - batch_start_time)
            mean_batch_time = sum(batch_times[-8:]) / min(len(batch_times), 8)

            assert len(results) == 1, "assert len(models) == 1 before this"
            seq_lengths, output_tensor = results[0]
            # seq_lengths, input_tensor, output_tensor = results[0]
            embeddings.append(output_tensor.cpu().numpy())

            # Progress.
            if batch_index % 1 == 0:
                est_dataset_time = (batch_end_time - dataset_start_time) + \
                    (n_batches - batch_index - 1) * mean_batch_time
                samples_per_sec = len(data_loader.dataset) / est_dataset_time
                print_rank_0("batch %d / %d [%d] ... %.3f samples/sec [ 47b = %.1f node days ] ... %s" % (
                    batch_index,
                    n_batches,
                    seq_lengths.max().item(),
                    samples_per_sec,
                    (47e9 / samples_per_sec) / 16 / (24 * 3600),
                    get_mem_stats_str(),
                ))
                # >>>
                # if batch_index == 6:
                #     pax(0, {
                #         "seq_lengths" : str(seq_lengths),
                #         "input_tensor" : str(input_tensor.shape),
                #         "output_tensor" : str(output_tensor.shape),
                #     })
                # <<<

    return np.concatenate(embeddings, axis = 0)


# def embed_blocks(args, models, prefix, dataset, missing_embedding_blocks):
# def embed_blocks(args, models, workdir, dataset, missing_embedding_blocks):
# def embed_blocks(args, models, prefix, workdir, dataset,
def embed_blocks(models, prefix, workdir, dataset, missing_embedding_blocks):

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
            print_rank_0("embed '%s' block %d / %d ... %s." % (
                prefix,
                block_index,
                len(missing_embedding_blocks),
                # workdir + "/" + block_info["path"],
                block_info["path"],
            ))

            # Data loader.
            data_loader = get_block_data_loader(dataset,*block_info["range"])

            # Embed block.
            embeddings = embed_batches(models, data_loader)

            # Save embeddings.
            f = h5py.File(block_info["path"], "w")
            f.create_dataset("data", data = embeddings)
            f.close()

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()


def embed_text_dataset(models, prefix, workdir, text_dataset, block_size):

    # Dataset workdir.
    os.makedirs(workdir, exist_ok = True)

    # Bert embedding dataset.
    embedding_dataset = BertEmbeddingDataset(text_dataset)

    # Missing embedding blocks (stored on disk).
    missing_embedding_blocks = get_missing_embedding_blocks(workdir,
                                                            embedding_dataset,
                                                            block_size)

    # pax(0, {
    #     "missing_embedding_blocks" : missing_embedding_blocks,
    #     "workdir" : workdir,
    #     "text_dataset" : text_dataset,
    #     "embedding_dataset" : embedding_dataset,
    # })

    # Prevent missing file race condition.
    torch.distributed.barrier()

    # Embed batches.
    embed_blocks(models, prefix, workdir, embedding_dataset,
                 missing_embedding_blocks)


# def embed_text_datasets(args, text_dataset_map):
def embed_text_datasets(text_dataset_map, block_size):

    # Load model.
    models, optimizer, opt_param_scheduler = \
        setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

    # Embed each (i.e., full, sampled) dataset.
    for prefix, info in text_dataset_map.items():
        print_rank_0(" > embed '%s' dataset ... %d samples." %
                     (prefix, len(info["data"])))
        embed_text_dataset(models, prefix,
                           info["embed_dir"], info["data"],
                           block_size)

