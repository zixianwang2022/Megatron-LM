# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import hashlib
import os
import torch

from megatron import get_retro_args, print_rank_0
# >>>
# from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.gpt_dataset import build_train_valid_test_datasets \
    as build_gpt_train_valid_test_datasets
# <<<
from megatron.training import (
    # >>>
    # build_train_valid_test_data_loaders,
    build_train_valid_test_datasets as build_pretraining_train_valid_test_datasets,
    # <<<
    update_train_iters,
)
from tools.retro.db.utils import get_indexed_dataset_infos
from tools.retro.utils import get_num_chunks_per_sample

from .utils import get_query_workdir


class ChunkDataset(torch.utils.data.Dataset):
    '''Pretraining chunk dataset wraps a standard GPT dataset.

    This dataset conceptually divides each sample (e.g., length 2048)
    into chunks (e.g., length 64) and restructures them into a list of
    chunks (e.g., length num_samples * num_chunks_per_sample).
    '''

    def __init__(self, sample_dataset, chunk_length):

        super().__init__()

        self.sample_dataset = sample_dataset

        self.chunk_length = chunk_length
        self.n_chunks_per_sample = get_num_chunks_per_sample()
        self.n_samples = len(sample_dataset)
        self.n_chunks = self.n_samples * self.n_chunks_per_sample

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):

        # Convert global chunk index to global sample index & local chunk index.
        sample_idx = idx // self.n_chunks_per_sample
        chunk_idx = idx % self.n_chunks_per_sample

        # Extract sample data.
        sample = self.sample_dataset[sample_idx]
        sample_token_ids = sample["text"]
        sample_doc_ids = sample["doc_ids"]

        # Chunk start/end token idxs.
        token_start_idx = chunk_idx * self.chunk_length
        token_end_idx = token_start_idx + self.chunk_length
        chunk_token_ids = sample_token_ids[token_start_idx:token_end_idx]

        # Sample.
        return {
            "doc_ids" : sample_doc_ids,
            "text" : chunk_token_ids,
        }


def verify_indexed_dataset_order():
    '''Verify pretraining order same as DB order.'''

    args = get_retro_args()

    # DB dataset prefixes.
    db_indexed_dataset_infos = get_indexed_dataset_infos()
    db_prefixes = [ info["prefix"] for info in db_indexed_dataset_infos ]

    # Verify order & prefixes.
    assert len(args.data_path) >= 2, "blendable dataset supported only."
    pretraining_prefixes = args.data_path[1:None:2]

    if len(db_prefixes) != len(pretraining_prefixes):
        raise Exception("inconsistent dataset count between db & pretraining.")
    if db_prefixes != pretraining_prefixes:
        raise Exception("inconsistent dataset order between db & pretraining.")


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    args = get_retro_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_gpt_train_valid_test_datasets(
        data_prefix=args.retro_gpt_data_path,
        data_impl=args.retro_gpt_data_impl,
        splits_string=args.retro_gpt_split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.retro_gpt_seq_length,
        seed=args.retro_gpt_seed,
        skip_warmup=(not args.retro_gpt_mmap_warmup),
        global_batch_size=args.retro_gpt_global_batch_size,
        eval_interval=args.retro_gpt_eval_interval,
        eval_iters=args.retro_gpt_eval_iters,
        return_doc_ids=args.retro_return_doc_ids)
    print_rank_0("> finished creating pretrained GPT datasets ...")

    # >>>
    # from lutil import pax
    # pax({
    #     "train_ds" : "%d, %s" % (len(train_ds), 0), # train_ds.desc_hash),
    #     "valid_ds" : "%d, %s" % (len(valid_ds), 0), # valid_ds.desc_hash),
    #     "test_ds" : test_ds,
    # })
    # <<<

    return train_ds, valid_ds, test_ds


# >>>
# def get_chunk_dataset_map():
#     '''Get train, valid, test chunk datasets.'''

#     args = get_retro_args()

#     # Update train iters.
#     update_train_iters(args)

#     args.iteration = 0
#     args.consumed_train_samples = 0

#     # Verify indexed dataset order.
#     verify_indexed_dataset_order()

#     # Datasets.
#     print_rank_0(" > data loader.")
#     train_data_loader, valid_data_loader, test_data_loader \
#         = build_train_valid_test_data_loaders(
#             train_valid_test_datasets_provider)

#     data_loader_map = {
#         "train" : train_data_loader,
#         "valid" : valid_data_loader,
#         "test" : test_data_loader,
#     }

#     # >>>
#     # from lutil import pax
#     # train_loader = data_loader_map["train"]
#     # train_ds = train_loader.dataset
#     # pax({
#     #     "train_loader" : train_loader,
#     #     "train_ds" : train_ds,
#     #     "train_ds / datasets" : train_ds.datasets,
#     #     "train_ds / datasets / 0" : train_ds.datasets[0],
#     # })
#     # <<<

#     # Info dict.
#     workdir = get_query_workdir()
#     dataset_map = {
#         key : {
#             "neighbor_dir" : os.path.join(
#                 workdir,
#                 os.path.basename(loader.dataset.datasets[0].index_prefix),
#             ),
#             "data" : ChunkDataset(loader.dataset, args.retro_gpt_chunk_length),
#         }
#         for key, loader in data_loader_map.items() if loader
#     }

#     # >>>
#     from lutil import pax
#     pax({"dataset_map": dataset_map})
#     # <<<

#     return dataset_map
# +++
def get_chunk_dataset_map():
    '''Get train, valid, test chunk datasets.'''

    args = get_retro_args()

    # Update train iters.
    update_train_iters(args)

    args.iteration = 0
    args.consumed_train_samples = 0

    # Verify indexed dataset order.
    verify_indexed_dataset_order()

    # Datasets.
    print_rank_0(" > datasets.")
    train_ds, valid_ds, test_ds = build_pretraining_train_valid_test_datasets(
        train_valid_test_datasets_provider)

    # >>>
    # from lutil import pax
    # pax({
    #     "train_ds / datasets" : train_ds.datasets,
    #     "train_ds / datasets / 0" : train_ds.datasets[0],
    #     "train_ds / len" : len(train_ds),
    #     "valid_ds / len" : len(valid_ds),
    #     "test_ds" : test_ds,
    # })
    # <<<

    sample_dataset_map = {
        "train" : train_ds,
        "valid" : valid_ds,
        "test" : test_ds,
    }

    # >>>
    def get_dataset_hash(dataset):
        hashes = ",".join([ d.desc_hash for d in dataset.datasets ])
        return hashlib.md5(hashes.encode()).hexdigest()
    # <<<

    # Info dict.
    workdir = get_query_workdir()
    chunk_dataset_map = {
        key : {
            "neighbor_dir" : os.path.join(
                workdir,
                # >>>
                # os.path.basename(sample_ds.datasets[0].index_prefix),
                os.path.basename(key + "_" + get_dataset_hash(sample_ds)),
                # <<<
            ),
            "data" : ChunkDataset(sample_ds, args.retro_gpt_chunk_length),
        }
        for key, sample_ds in sample_dataset_map.items() if sample_ds
    }

    # >>>
    from lutil import pax
    # pax({"chunk_dataset_map": chunk_dataset_map})
    pax(chunk_dataset_map)
    # <<<

    return dataset_map
# <<<
