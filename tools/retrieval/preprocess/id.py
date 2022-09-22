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

# """Pretrain GPT"""
# from functools import partial
import h5py
import joblib
# import multiprocessing
import numpy as np
# import time
import torch
from tqdm import tqdm

from megatron import (
    get_args,
    get_timers,
    get_tokenizer,
    initialize_megatron,
    mpu,
    print_rank_0,
)
# from megatron.data.blendable_dataset import BlendableDataset
# from megatron.model import GPTModel, ModelType
from megatron.training import (
    build_train_valid_test_data_iterators,
    # pretrain,
    print_datetime,
    update_train_iters,
)
# from megatron.utils import get_ltor_masks_and_position_ids
# from megatron.utils import average_losses_across_data_parallel_group

# _TRAIN_START_TIME = time.time()

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

def get_batch_for_preprocess(data_iterator):
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    tokens_ = data['text']
    tokens = tokens_[:, :-1].contiguous()
    # pax(0, {"data": data, "doc_ids": data["doc_ids"]})
    return tokens, [doc.item() for doc in data["doc_ids"]], data['idx']

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


# def train_valid_test_datasets_provider(train_val_test_num_samples):
# def datasets_provider(train_val_test_num_samples):
def train_valid_test_dataset_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    if 1:
        from megatron.data.gpt_dataset import build_train_valid_test_datasets
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    else:
        from megatron.data.dataset_utils import build_train_valid_test_datasets
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            max_seq_length=args.seq_length,
            masked_lm_prob=args.mask_prob,
            short_seq_prob=args.short_seq_prob,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    # pax(0, {
    #     "train_ds" : train_ds,
    #     "lens" : [ len(ds) for ds in [ train_ds, valid_ds, test_ds ] ],
    # })
    print_rank_0("> finished creating pretrained GPT datasets ...")

    # train_ds = train_ds1
    # train_ds = BlendableDataset([train_ds1, train_ds2], [args.weight, 1 - args.weight])

    return train_ds, valid_ds, test_ds


def save_document_ids(retrieval_args, timer):

    assert torch.distributed.get_rank() == 0, "single process operation."

    args = get_args()
    timers = get_timers()

    # # Adjust the startup time so it reflects the largest value.
    # # This will be closer to what scheduler will see (outside of
    # # image ... launches.
    # global _TRAIN_START_TIME
    # start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    # torch.distributed.all_reduce(start_time_tensor,
    #                              op=torch.distributed.ReduceOp.MIN)
    # _TRAIN_START_TIME = start_time_tensor.item()
    # print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
    #     time.time() - _TRAIN_START_TIME))
    # print_datetime('after megatron is initialized')

    # >>>
    update_train_iters(args)
    # <<<

    # pax({"args": args})

    args.iteration = args.embed_start_index
    args.consumed_train_samples = args.embed_start_index  # consumed samples == iterations (bs=1)

    # pax(0, {"args": args})

    # Model, optimizer, and learning rate.
    # Data stuff.
    timers('train/valid/test-data-iterators-setup').start()

    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    # data_iterators = build_train_valid_test_data_iterators(datasets_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # # >>>
    # args.train_samples = len(train_data_iterator)
    # update_train_iters(args)
    # # <<<

    # pax(0, {
    #     "data_iterators" : [ len(i) if i else None for i in [
    #         train_data_iterator,
    #         valid_data_iterator,
    #         test_data_iterator,
    #     ]],
    #     "args.train_samples" : args.train_samples,
    #     "args.train_iters" : args.train_iters,
    #     "args.do_train" : args.do_train,
    # })

    # Print setup timing.
    print_rank_0('done with setup ...')
    print_rank_0('training ...')

    data_path = args.data_path[0]
    print(args.data_path)

    if args.neighbors_path:
        data_path = args.neighbors_path

    # pax({"data_path": data_path})

    if args.add_offset_doc_ids:
        output_file = data_path + f"_start_{args.embed_start_index}_end_{args.embed_end_index}_ns_{args.train_iters}_sl_{args.seq_length}_seed_{args.seed}_with_offset.tokens.h5py"
        doc_ids_file = data_path + f"_start_{args.embed_start_index}_end_{args.embed_end_index}_ns_{args.train_iters}_sl_{args.seq_length}_seed_{args.seed}_with_offset.doc_ids.pkl"
    else:
        output_file = data_path + f"_start_{args.embed_start_index}_end_{args.embed_end_index}_ns_{args.train_iters}_sl_{args.seq_length}_seed_{args.seed}.tokens.h5py"
        doc_ids_file = data_path + f"_start_{args.embed_start_index}_end_{args.embed_end_index}_ns_{args.train_iters}_sl_{args.seq_length}_seed_{args.seed}.doc_ids.pkl"
    print("Dumping: ", output_file)
    print("Dumping: ", doc_ids_file)

    # pax(0, {
    #     "output_file" : output_file,
    #     "doc_ids_file" : doc_ids_file,
    # })

    output = np.zeros((args.embed_end_index - args.embed_start_index, 2048), dtype='uint16')
    doc_ids_list = []

    if args.do_train and args.train_iters > 0:
        for iteration in tqdm(range(args.embed_start_index, args.embed_end_index)):
            timers('batch-generator').start()
            tokens, doc_ids, data_idx = get_batch_for_preprocess(train_data_iterator)
            # pax(0, {
            #     "tokens" : tokens,
            #     "doc_ids" : doc_ids,
            #     "data_idx" : data_idx,
            # })
            if iteration - args.embed_start_index < 10:
                print(tokens, doc_ids, data_idx)
            output[iteration - args.embed_start_index] = tokens[0].numpy()
            doc_ids_list.append(doc_ids)
            timers('batch-generator').stop()

        print_datetime('after training is done')

    # >>>
    pax({
        "output" : str(output.shape),
        "doc_ids_list" : doc_ids_list,
    })
    # raise Exception("ready to dump?")
    # <<<

    output_file = h5py.File(output_file, "w")
    output_file.create_dataset("tokens", data=output)
    output_file.close()

    joblib.dump(doc_ids_list, doc_ids_file)


# if __name__ == "__main__":
#     retrieve_dataset(train_valid_test_datasets_provider,
#                      args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
#                      extra_args_provider=add_validation_args)

# eof
