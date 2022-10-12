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
# from megatron.data.data_samplers import MegatronPretrainingSampler
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.model import BertModel, ModelType
from megatron.schedules import get_forward_backward_func
from megatron.training import (
    setup_model_and_optimizer,
)
# from pretrain_bert import (
#     # forward_step as forward_step_func,
#     get_batch,
#     model_provider,
#     # train_valid_test_datasets_provider,
# )

# from ..preprocess.utils import get_sampled_chunk_index_path
from ..preprocess.utils import get_chunk_index_path_map
from .chunk_dataset import BertChunkDataset
from .long_bert_chunks import print_longest_bert_chunks

# >>>
from lutil import pax, print_seq
# <<<


# def get_n_chunks(args):
#     chunk_index_path = get_sampled_chunk_index_path(args.retrieval_workdir)
#     f = h5py.File(chunk_index_path, "r")
#     n_chunks = len(f["chunks_valid"])
#     f.close()
#     return n_chunks
# def get_n_chunks(args):
#     chunk_index_path_map = get_chunk_index_path_map(args.retrieval_workdir)
#     pax(0, {"chunk_index_path_map": chunk_index_path_map})
#     f = h5py.File(chunk_index_path, "r")
#     n_chunks = len(f["chunks_valid"])
#     f.close()
#     return n_chunks


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


# def get_missing_embedding_blocks(args, workdir, data_loader):
# def get_missing_embedding_blocks(args, workdir, n_chunks):
# def get_missing_embedding_blocks(args, workdir, prefix, dataset_info):
def get_missing_embedding_blocks(args, workdir, dataset_info):

    # n_chunks = len(data_loader)
    # n_chunks = len(dataset_info["chunk_index"])
    n_chunks = len(dataset_info["dataset"])

    # pax(0, {"workdir": workdir, "n_chunks": n_chunks})

    # Block ranges.
    block_size = args.retrieval_block_size
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
            # "%s_%s-%s.hdf5" % (prefix, *[ str(i).zfill(n_digits) for i in r ]),
        )
    } for r in block_ranges]

    # pax(0, {"all_block_items": all_block_items})

    # Delete corrupt files.
    if torch.distributed.get_rank() == 0:
        existing_block_paths = [item["path"]
                                for item in all_block_items
                                if os.path.exists(item["path"])]
        # pax({"existing_block_paths": existing_block_paths})
        pbar = tqdm(existing_block_paths)
        for index, path in enumerate(pbar):
            # print_rank_0(" > verifying embedding block %d / %d, '%s'." % (
            #     index,
            #     len(existing_block_paths),
            #     os.path.basename(path),
            # ))
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
    # print_seq("my inital ranges are %s." % ", ".join(str(i["range"]) for i in rank_missing_block_items[:3]))
    # pax(0, {
    #     "rank_missing_block_items" : rank_missing_block_items,
    #     "all_block_items / len" : len(all_block_items),
    #     "missing_block_items / len" : len(missing_block_items),
    #     # "missing_block_items / 0" : missing_block_items[0],
    #     "rank_missing_block_items / len" : len(rank_missing_block_items),
    #     "workdir" : workdir,
    #     # "data_loader / len" : len(data_loader),
    #     "n_chunks" : n_chunks,
    #     "block_size" : block_size,
    # })
    # print_seq("my start/end ranges [ %d ] ... %s, %s." % (
    #     len(rank_missing_block_items),
    #     str(rank_missing_block_items[0]["range"]),
    #     str(rank_missing_block_items[-1]["range"]),
    # ))
    # <<<

    return rank_missing_block_items

    
# def get_missing_embedding_block_map(args, workdir, dataset_info_map):

#     missing_block_map = {
#         key : get_missing_embedding_blocks(args, workdir, key, dataset_info)
#         for key, dataset_info in dataset_info_map.items()
#     }

#     pax(0, {"missing_block_map": missing_block_map})

#     return missing_block_map

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
# def get_shared_dataset_info(args):

#     # Load dataset metadata.
#     with open(os.path.join(args.retrieval_workdir, "order.json")) as f:
#         data_metas = json.load(f)

#     # Token datasets.
#     indexed_datasets = \
#         [ make_indexed_dataset(m["prefix"], "mmap", True) for m in data_metas ]

#     # Chunk index.
#     chunk_index_path = get_sampled_chunk_index_path(args.retrieval_workdir)
#     f = h5py.File(chunk_index_path, "r")
#     dataset_offsets = np.copy(f["dataset_offsets_valid"])
#     chunk_index = np.copy(f["chunks_valid"])
#     f.close()

#     dataset_ids = []
#     for i in range(len(dataset_offsets) - 1):
#         dataset_ids.append([i] * (dataset_offsets[i+1] - dataset_offsets[i]))
#     dataset_ids = [ i for ii in dataset_ids for i in ii ]

#     return {
#         "data_metas" : data_metas,
#         "indexed_datasets" : indexed_datasets,
#         # "dataset_offsets" : dataset_offsets,
#         "dataset_ids" : dataset_ids,
#         "chunk_index" : chunk_index,
#     }
# def get_shared_dataset_info_map(args):
def get_dataset_info_map(args):

    # Load dataset metadata.
    with open(os.path.join(args.retrieval_workdir, "order.json")) as f:
        data_metas = json.load(f)

    # Token datasets.
    indexed_datasets = \
        [ make_indexed_dataset(m["prefix"], "mmap", True) for m in data_metas ]

    # Chunk index.
    chunk_index_path_map = get_chunk_index_path_map(args.retrieval_workdir)
    dataset_info_map = {} # key: {} for key in chunk_index_path_map}
    for key, chunk_index_path in chunk_index_path_map.items():

        # Load chunk index.
        f = h5py.File(chunk_index_path, "r")
        dataset_offsets = np.copy(f["dataset_offsets_valid"])
        chunk_index = np.copy(f["chunks_valid"])
        f.close()

        # if key == "sampled":
        #     pax(0, {
        #         "data_metas" : [ d["prefix"] for d in data_metas ],
        #         "dataset_offsets" : str(dataset_offsets),
        #     })

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
            max_chunk_length = args.retrieval_chunk_length,
            max_model_seq_length = args.seq_length,
            masked_lm_prob = args.mask_prob,
            seed = args.seed,
        )

        # dataset_info_map[key] = {
        #     "data_metas" : data_metas,
        #     "indexed_datasets" : indexed_datasets,
        #     # "dataset_offsets" : dataset_offsets,
        #     "dataset_ids" : dataset_ids,
        #     "chunk_index" : chunk_index,
        # }
        dataset_info_map[key] = {
            "dataset" : dataset,
        }

    # pax(0, {
    #     "dataset_info_map" :
    #     {k:str(v) for k,v in dataset_info_map.items()},
    # })
    # pax(0, dataset_info_map)

    return dataset_info_map


# def get_block_data_loader(args, shared_dataset_info, chunk_start_id, chunk_end_id):

#     # pax(0, {"shared_dataset_info": shared_dataset_info})

#     # Chunk dataset.
#     # t = time.time()
#     dataset = BertChunkDataset(
#         indexed_datasets = shared_dataset_info["indexed_datasets"],
#         dataset_ids = shared_dataset_info["dataset_ids"][chunk_start_id:chunk_end_id],
#         chunk_index = shared_dataset_info["chunk_index"][chunk_start_id:chunk_end_id],
#         max_chunk_length = args.retrieval_chunk_length,
#         # max_seq_len = args.seq_length,
#         max_model_seq_length = args.seq_length,
#         # micro_batch_size = args.micro_batch_size,
        
#         masked_lm_prob = args.mask_prob,
#         seed = args.seed,
        
#         # >>>
#         # binary_head = args.bert_binary_head,
#         # binary_head = False, # allows len(segments) == 1
#         # <<<
#     )

#     # t = time.time() - t
#     # print_seq("chunk dataset, %.3f sec." % t)

#     # Megatron sampler.
#     # >>>
#     # batch_sampler = MegatronPretrainingSampler(
#     #     # total_samples = len(chunk_index),
#     #     total_samples = chunk_end_id - chunk_start_id,
#     #     consumed_samples = 0,
#     #     micro_batch_size = args.micro_batch_size,
#     #     data_parallel_rank = mpu.get_data_parallel_rank(),
#     #     data_parallel_size = mpu.get_data_parallel_world_size())
#     # +++
#     from torch.utils.data import BatchSampler, SequentialSampler
#     batch_sampler = BatchSampler(
#         sampler = SequentialSampler(dataset),
#         batch_size = args.micro_batch_size,
#         drop_last = False,
#     )
#     # pax(0, {"batch_sampler": batch_sampler})
#     # +++

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
def collate_batch(samples):
    """Collate samples of various lengths.

    This collate function handles samples with various sequence lengths, by
    padding 'text' arrays with pad_id, and other arrays with 0.
    """

    n_samples = len(samples)
    keys = list(samples[0].keys())
    tokenizer = get_tokenizer()
    # pax(0, {"tokenizer": tokenizer})

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

    # pax(0, {"batch": batch})

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


# def loss_func(loss_mask, sentence_order, output_tensor, non_loss_data):
#     assert non_loss_data
#     return output_tensor


# def forward_step(data_iterator, model):
#     """Forward step."""
#     args = get_args()

#     # Get the batch.
#     tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
#         data_iterator)

#     # >>>
#     # pax(0, {
#     #     "data_iterator" : data_iterator,
#     #     "tokens" : tokens,
#     #     "types" : types,
#     #     "sentence_order" : sentence_order,
#     #     "loss_mask" : loss_mask,
#     #     "lm_labels" : lm_labels,
#     #     "padding_mask" : padding_mask,
#     # })
#     # <<<

#     if not args.bert_binary_head:
#         types = None

#     # Forward pass through the model.
#     output_tensor = model(tokens, padding_mask, tokentype_ids=types,
#                           lm_labels=lm_labels)

#     # >>>
#     # print(tokens)
#     # pax(0, {
#     #     "model" : model,
#     #     "tokens" : tokens,
#     #     "output_tensor" : output_tensor,
#     # })
#     # <<<

#     return output_tensor, partial(loss_func, loss_mask, sentence_order)


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
            # output_tensors = forward_backward_func(
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

            # pax(0, {"seq_lengths": seq_lengths, "output_tensor": output_tensor})

            # Progress.
            est_dataset_time = (batch_end_time - dataset_start_time) + \
                (n_batches - batch_index - 1) * mean_batch_time
            samples_per_sec = len(data_loader.dataset) / est_dataset_time
            print_rank_0("batch %d / %d [%d] ... %.3f samples/sec [ 47b = %.1f node days ]." % (
                batch_index,
                n_batches,
                # data_loader.dataset.batch_chunk_lens[batch_index],
                seq_lengths.max().item(),
                samples_per_sec,
                (47e9 / samples_per_sec) / 16 / (24 * 3600),
            ))

            # raise Exception("hi.")

    return np.concatenate(embeddings, axis = 0)


# def embed_blocks(args, models, shared_dataset_info, missing_embedding_blocks):
def embed_blocks(args, models, prefix, dataset_info, missing_embedding_blocks):

    print_seq("%d blocks." % len(missing_embedding_blocks))

    # Iterate blocks.
    for block_index, block_info in enumerate(missing_embedding_blocks):

        # Missing block lists are extended with None to have equal-length
        # lists. Skip the Nones.
        if block_info is not None:

            print_rank_0("embed '%s' block %d / %d ... %s." % (
                prefix,
                block_index,
                len(missing_embedding_blocks),
                os.path.basename(block_info["path"]),
            ))

            # Data loader.
            data_loader = get_block_data_loader(args,
                                                # dataset_info,
                                                dataset_info["dataset"],
                                                *block_info["range"])

            embeddings = embed_batches(args, models, data_loader)

            f = h5py.File(block_info["path"], "w")
            f.create_dataset("data", data = embeddings)
            f.close()

            # pax(0, {
            #     "embeddings" : embeddings,
            #     "data_loader" : data_loader,
            #     "data_loader / len" : len(data_loader),
            #     "block_info" : block_info,
            # })

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()

    # >>>
    # torch.distributed.barrier()
    # print_seq("finished embedding '%s'." % prefix)
    # <<<


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# def embed_chunks(args, timer):

#     # print_seq("i am data rank %d." % mpu.get_data_parallel_rank())

#     # Embedding workdir.
#     workdir = os.path.join(args.retrieval_workdir, "embed")
#     os.makedirs(workdir, exist_ok = True)

#     # Load model.
#     models, optimizer, opt_param_scheduler = \
#         setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

#     # Share dataset info (indexed datasets, chunk index, etc.).
#     # t = time.time()
#     shared_dataset_info = get_shared_dataset_info(args)
#     # t = time.time() - t
#     # print_seq("shared dataset info, time = %.3f." % t)

#     # pax(0, {"shared_dataset_info_map": shared_dataset_info_map})

#     # >>>
#     # bert_chunk_lens = [ a[3] for a in shared_dataset_info["chunk_index"] ]
#     # pax(0, {
#     #     "shared_dataset_info" : shared_dataset_info,
#     #     "bert_chunks_lens" : bert_chunk_lens[:20],
#     #     "bert_chunks_lens / len" : len(bert_chunk_lens),
#     #     "mean" : np.mean(bert_chunk_lens),
#     #     "var" : np.var(bert_chunk_lens),
#     # })
#     # <<<

#     # >>>
#     # print_longest_bert_chunks(args, shared_dataset_info)
#     # raise Exception("hi.")
#     # <<<

#     # >>>
#     # from .test_huggingface import test_huggingface
#     # test_huggingface(args, shared_dataset_info, timer)
#     # raise Exception("hi.")
#     # <<<

#     # Missing embedding blocks (stored on disk).
#     n_chunks = get_n_chunks(args)
#     missing_embedding_blocks = \
#         get_missing_embedding_blocks(args, workdir, n_chunks) # data_loader)

#     # Prevent missing file race condition.
#     torch.distributed.barrier()

#     # pax(0, {
#     #     "missing_embedding_blocks" : missing_embedding_blocks,
#     #     "n_chunks" : n_chunks,
#     # })

#     # # >>>
#     # # Data loader.
#     # # t = time.time()
#     # data_loader = get_chunk_data_loader(args, data_metas, timer)
#     # # t = time.time() - t
#     # # print_seq("data_loader, %.3f sec." % t)
#     # # <<<

#     # pax(0, {
#     #     "data_metas" : data_metas,
#     #     # "data_loader" : data_loader,
#     #     "data_loader / len" : len(data_loader),
#     # })

#     # Embed batches.
#     # embed_batches(args, models, data_loader, missing_embedding_blocks)
#     embed_blocks(args, models, shared_dataset_info, missing_embedding_blocks)

#     raise Exception("unsort chunks.")
def embed_dataset_chunks(args, workdir, models, prefix, dataset_info):

    # Dataset workdir.
    workdir = os.path.join(workdir, prefix)
    os.makedirs(workdir, exist_ok = True)

    # Missing embedding blocks (stored on disk).
    missing_embedding_blocks = \
        get_missing_embedding_blocks(args, workdir, dataset_info)

    # pax(0, {"missing_embedding_blocks": missing_embedding_blocks})

    # Prevent missing file race condition.
    torch.distributed.barrier()

    # Embed batches.
    embed_blocks(args, models, prefix, dataset_info, missing_embedding_blocks)

    raise Exception("unsort chunks.")

def embed_chunks(args, timer):

    # Embedding workdir.
    workdir = os.path.join(args.retrieval_workdir, "embed")
    os.makedirs(workdir, exist_ok = True)

    # Load model.
    models, optimizer, opt_param_scheduler = \
        setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

    # Dataset infos (indexed datasets, chunk index, etc.).
    dataset_info_map = get_dataset_info_map(args)

    # Embed each (i.e., full, sampled) dataset.
    for prefix, dataset_info in dataset_info_map.items():
        print_rank_0(" > embed '%s' chunks. [ count %d ]" %
                     # (prefix, len(dataset_info["chunk_index"])))
                     (prefix, len(dataset_info["dataset"])))
        embed_dataset_chunks(args, workdir, models, prefix, dataset_info)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# eof
