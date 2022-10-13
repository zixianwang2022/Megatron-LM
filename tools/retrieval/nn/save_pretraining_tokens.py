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

? ? ? unnecessary script ? ? ?

# from functools import partial
# import h5py
# import multiprocessing
# import numpy as np
import os
# import time
# import torch
from tqdm import tqdm

from megatron import (
    get_args,
    # get_timers,
    # get_tokenizer,
    # initialize_megatron,
    # mpu,
    print_rank_0,
)
# from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.gpt_dataset import build_train_valid_test_datasets
# from megatron.model import GPTModel, ModelType
from megatron.training import (
    # pretrain,
    # print_datetime,
    build_train_valid_test_data_iterators,
    update_train_iters,
)
# from megatron.utils import get_ltor_masks_and_position_ids
# from megatron.utils import average_losses_across_data_parallel_group

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


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def get_batch_for_preprocess(data_iterator):
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    pax(0, {"data": data})
    tokens_ = data['text']
    tokens = tokens_[:, :-1].contiguous()
    return tokens, [doc.item() for doc in data["doc_ids"]], data['idx']

def get_batch_for_preprocess_by_data(data):
    tokens_ = data['text']
    tokens = tokens_[:, :-1].contiguous()
    return tokens, [doc.item() for doc in data["doc_ids"]]



def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


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
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating pretrained GPT datasets ...")

    # train_ds = train_ds1
    # train_ds = BlendableDataset([train_ds1, train_ds2], [args.weight, 1 - args.weight])

    return train_ds, valid_ds, test_ds


def save_pretraining_tokens(args, timer):

    # Workdir.
    workdir = os.path.join(args.retrieval_workdir, "nn")
    os.makedirs(workdir, exist_ok = True)

    # Update train iters.
    update_train_iters(args)

    # pax(0, {"args": args})

    args.start = 0
    args.end = args.train_samples
    args.iteration = args.start
    args.consumed_train_samples = args.start  # consumed samples == iterations (bs=1)
    # args.iteration = 0
    # args.consumed_train_samples = 0

    # Data stuff.
    print_rank_0(" > data iterators.")
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
            train_valid_test_datasets_provider)

    # # Print setup timing.
    # print_rank_0('done with setup ...')
    # print_rank_0('training ...')

    # data_path = args.data_path[0]
    # print(args.data_path)

    # exit(0)

    # if args.neighbors_path:
    #     data_path = args.neighbors_path

    # if args.add_offset_doc_ids:
    #     output_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}_seed_{args.seed}_with_offset.tokens.h5py"
    #     doc_ids_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}_seed_{args.seed}_with_offset.doc_ids.pkl"
    # else:
    #     output_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}_seed_{args.seed}.tokens.h5py"
    #     doc_ids_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}_seed_{args.seed}.doc_ids.pkl"

    # token_path = os.path.join(workdir, f"tokens_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}_seed_{args.seed}.h5py")
    doc_ids_path = os.path.join(workdir, f"doc_ids_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}_seed_{args.seed}.pkl")

    # print("Dumping: ", token_path)
    print("Dumping: ", doc_ids_path)

    # import h5py
    # output = np.zeros((args.end - args.start, 2048), dtype='uint16')
    doc_ids_list = []

    if args.do_train and args.train_iters > 0:
        for iteration in tqdm(range(args.start, args.end)):
            # timers('batch-generator').start()
            tokens, doc_ids, data_idx = get_batch_for_preprocess(train_data_iterator)
            # if iteration - args.start < 10:
            #     print(tokens, doc_ids, data_idx)
            pax(0, {"tokens": tokens, "doc_ids": doc_ids, "data_idx": data_idx})
            output[iteration - args.start] = tokens[0].numpy()
            doc_ids_list.append(doc_ids)
            # timers('batch-generator').stop()

        print_datetime('after training is done')

    output_file = h5py.File(output_file, "w")
    output_file.create_dataset("tokens", data=output)
    output_file.close()
    import joblib
    joblib.dump(doc_ids_list, doc_ids_file)

