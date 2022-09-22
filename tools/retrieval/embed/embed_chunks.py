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
import h5py
import numpy as np
import os
import torch

from megatron import (
    # get_args,
    # get_tokenizer,
    mpu,
    # print_rank_0,
)
from megatron.data.data_samplers import MegatronPretrainingSampler
# from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
# from megatron.initialize import initialize_megatron
from megatron.model import (
    # BertModel,
    ModelType,
)
# from megatron.schedules import get_forward_backward_func
from megatron.training import (
    # build_train_valid_test_data_iterators,
    setup_model_and_optimizer,
)
from pretrain_bert import (
    # forward_step as forward_step_func,
    # get_batch,
    model_provider,
    # train_valid_test_datasets_provider,
)
from tools.retrieval.preprocess.utils import (
    get_chunk_index_path,
    get_chunk_embedding_path,
)

from .chunk_dataset import GPTChunkDataset

# >>>
from lutil import pax, print_seq
# <<<

# ... [no] ... def initialize_megatron_for_embedding(retrieval_args):

#     # pretrain(train_valid_test_datasets_provider, model_provider,
#     #          ModelType.encoder_or_decoder,
#     #          forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})

#     initialize_megatron(
#         ignore_unknown_args = True,
#         args_defaults = {
#             # "tokenizer_type": "BertWordPieceCase",
#             "tokenizer_type": "BertWordPieceLowerCase",
#             "data_path" : [ retrieval_args.token_data_path ],
#             "vocab_file" : retrieval_args.token_vocab_file,

#             # "tensor_model_parallel_size" : 2,
#             # "pipeline_model_parallel_size" : 2,
#             "num_layers" : 24,
#             "hidden_size" : 1024,
#             "num_attention_heads" : 16,
#             # "micro_batch_size" : 2,
#             "micro_batch_size" : 128, # *2, 128, 1024
#             # "global_batch_size" : 16,
#             "seq_length" : 512,
#             "max_position_embeddings" : 512,
#             "train_iters" : 1, # 1000000,
#             # "save" : $CHECKPOINT_PATH,
#             "load" : retrieval_args.bert_load_path,
#             # "data_impl" : mmap,
#             # "split" : 949,50,1,
#             # "distributed_backend" : nccl,
#             "lr" : 0.0001,
#             # "lr_decay_style" : linear,
#             # "min_lr" : 1.0e-5,
#             # "lr_decay_iters" : 990000,
#             # "weight_decay" : 1e-2,
#             # "clip_grad" : 1.0,
#             # "lr_warmup_fraction" : .01,
#             # "log_interval" : 100,
#             # "save_interval" : 10000,
#             # "eval_interval" : 1000,
#             # "eval_iters" : 10,
#             # "fp16" : True,
#         },
#     )

# ... [no] ... def model_provider(pre_process=True, post_process=True):
#     """Build the model."""

#     print_rank_0('building BERT model ...')

#     args = get_args()
#     num_tokentypes = 2 if args.bert_binary_head else 0
#     model = BertModel(
#         num_tokentypes=num_tokentypes,
#         add_binary_head=args.bert_binary_head,
#         parallel_output=True,
#         pre_process=pre_process,
#         post_process=post_process)
#         # post_process=False)

#     # >>>
#     # print(model)
#     # print(model.lm_head)
#     # pax(0, {})
#     # <<<

#     return model

def loss_func(loss_mask, sentence_order, output_tensor, non_loss_data):
    assert non_loss_data
    # >>>
    pax(0, {
        "output_tensor" : output_tensor,
        "non_loss_data" : non_loss_data,
    })
    # <<<
    return output_tensor






    raise Exception("hi.")
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}

    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    # raise Exception("hi.")
    """Forward step."""
    args = get_args()
    # timers = get_timers()

    # Get the batch.
    # timers('batch-generator').start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    # timers('batch-generator').stop()

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
    # pax(0, {
    #     "model" : model,
    #     "tokens" : tokens,
    #     "output_tensor" : output_tensor,
    # })
    # <<<

    return output_tensor, partial(loss_func, loss_mask, sentence_order)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=64, # args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds

# def embed_chunks():
def embed_batches(models, data_iterator):

    for m in models:
        m.eval()

    forward_backward_func = get_forward_backward_func()
    # data_iterator = train_data_iterator
    with torch.no_grad():

        for batch_index in range(len(data_iterator)):

            print_rank_0("bach %d / %d." % (batch_index, len(data_iterator)))

            loss_dicts = forward_backward_func(
                forward_step, # _func,
                data_iterator,
                models,
                optimizer = None,
                timers = None,
                forward_only = True,
                collect_non_loss_data = True,
            )

            pax(0, {
                **{"loss_dicts / %d" % i : d for i, d in enumerate(loss_dicts)},
            })

def get_chunk_data_loader(args, models, data_prefix):

    # Token dataset.
    indexed_dataset = make_indexed_dataset(data_prefix, "mmap", True)

    # Chunk index.
    chunk_index_file = get_chunk_index_path(args, data_prefix)
    f = h5py.File(chunk_index_file, "r")
    eods = np.copy(f["eods"])
    chunk_index = np.copy(f["index"])
    f.close()

    # Chunk dataset.
    dataset = GPTChunkDataset(indexed_dataset, chunk_index, eods)

    # Megatron sampler.
    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(chunk_index),
        consumed_samples=0,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size())

    # Torch dataloader.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    # >>>
    pax({
        "chunk_index" : chunk_index,
        "chunk_dataset" : chunk_dataset,
        "dataset" : dataset,
        "batch_sampler" : batch_sampler,
        "data_loader" : data_loader,
    })
    # <<<

    return data_loader

def embed_chunks_single_dataset(args, models, data_prefix):

    data_loader = get_chunk_data_loader(args, models, data_prefix)

    pax({"data_loader": data_loader})

# def embed_chunks(retrieval_args, timer):
def embed_chunks(args, timer):

    # initialize_megatron_for_embedding(retrieval_args)

    # megatron_args = get_args()
    # megatron_args.bert_binary_head = False
    # megatron_args.model_type = ModelType.encoder_or_decoder

    models, optimizer, opt_param_scheduler = \
        setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    # for m in models:
    #     m.post_process = False

    # pax({
    #     "args" : args,
    #     "models" : models,
    # })

    # >>>
    # train_data_iterator, valid_data_iterator, test_data_iterator \
    #     = build_train_valid_test_data_iterators(
    #         train_valid_test_datasets_provider)
    # pax({"train_data_iterator": train_data_iterator})

    # embed_batches(models, train_data_iterator)

    # pax(0, {
    #     "retrieval_args" : retrieval_args,
    #     "megatron_args" : megatron_args,
    #     # "tokenizer" : tokenizer,
    #     "models" : models,
    #     # "optimizer" : optimizer,
    #     # "opt_param_scheduler" : opt_param_scheduler,
    #     # "train_data_iterator" : train_data_iterator,
    #     "train_data_iterator / len" : len(train_data_iterator),
    #     "valid_data_iterator / len" : len(valid_data_iterator),
    #     "test_data_iterator / len" : len(test_data_iterator),
    # })
    # <<<

    data_files = [ prefix.rstrip("/") + ".bin" for prefix in args.data_path ]
    data_files = [ path for path in data_files if os.path.exists(path) ]
    data_prefixes = [ os.path.splitext(f)[0] for f in data_files ]

    # pax({"data_prefixes": data_prefixes})

    for data_index, data_prefix in enumerate(data_prefixes):

        chunk_embedding_file = get_chunk_embedding_path(args, data_prefix)

        if os.path.exists(chunk_embedding_file):
            continue

        print(" > embedding chunks, dataset %d / %d ... '%s'." %
              (data_index, len(data_files), os.path.basename(data_prefix)))

        embeddings = embed_chunks_single_dataset(args, models, data_prefix)

        pax({"embeddings": embeddings})

    raise Exception("finished embedding?")

# eof
