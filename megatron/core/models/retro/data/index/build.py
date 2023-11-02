# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import os
import shutil
import torch
from tqdm import tqdm

# >>>
# from megatron import get_retro_args, print_rank_0
# from megatron.core.models.retro.data.db.utils import (
#     get_indexed_dataset_infos,
#     get_merged_sampled_dataset,
#     get_merged_train_dataset,
# )
# from megatron.core.models.retro.data.external_libs import h5py
# from megatron.core.models.retro.data.utils import GPTToTextDataset
# from tools.bert_embedding import DiskDataParallelBertEmbedder

from .factory import IndexFactory
# from .utils import (
#     get_training_data_block_dir,
#     get_training_data_block_paths,
#     get_training_data_merged_path,
#     get_training_data_root_dir,
# )
# <<<


##################################################
# Train index.
##################################################


def get_empty_index_path(env):
    '''Path of empty index.'''
    index = IndexFactory.get_index(env.config.retro_index_type)
    empty_index_path = index.get_empty_index_path()
    return empty_index_path


def get_block_nload(block_path, load_fraction):
    with h5py.File(block_path) as fi:
        return int(load_fraction * fi["data"].shape[0])


def merge_embedding_blocks(env):

    if torch.distributed.get_rank() != 0:
        return

    # Get block, merged paths.
    load_fraction = env.config.retro_index_train_load_fraction
    block_paths = get_training_data_block_paths(env)
    bin_path = get_training_data_merged_path(env)

    # Skip, if already built.
    if os.path.exists(bin_path):
        return

    # Merge blocks.
    with open(bin_path, "wb") as fo:
        byte_offset = 0
        for block_idx, block_path in \
            enumerate(tqdm(block_paths, "merge train embeddings")):
            with h5py.File(block_path) as fi:

                nload = get_block_nload(block_path, load_fraction)
                block = np.array(fi["data"][:nload], copy = False)

                fo.write(block.tobytes())

                byte_offset += block.size * block.itemsize
                fo.seek(byte_offset)


def embed_db(env):
    '''Embed DB chunks.

    Store chunks in blocks on disk. These blocks will later be merged into
    a single dataset for training the index.
    '''

    merged_train_data_path = get_training_data_merged_path(env)
    if os.path.exists(merged_train_data_path):
        return

    # Get db dataset.
    gpt_dataset = get_merged_sampled_dataset(env)
    text_dataset = GPTToTextDataset(gpt_dataset)

    # Embed dataset.
    embedder = DiskDataParallelBertEmbedder(env.config.retro_bert_batch_size,
                                            env.config.retro_bert_max_chunk_length,
                                            env.config.retro_block_size,
                                            env.config.bert_embedder_type)
    embedder.embed_text_dataset("index",
                                get_training_data_block_dir(env),
                                text_dataset)

    # Merge embeddings.
    merge_embedding_blocks(env)


def train_on_embeddings(env):
    '''Train index on embedded DB chunks.'''
    index = IndexFactory.get_index(env.config.retro_index_type)
    index.train()


def remove_embeddings(env):
    '''Remove embeddings after training.'''
    torch.distributed.barrier()
    if torch.distributed.get_rank() != 0:
        return
    empty_index_path = get_empty_index_path()
    assert os.path.isfile(empty_index_path)
    shutil.rmtree(get_training_data_root_dir(), ignore_errors=True)


def train_index(env):
    '''Train index on DB chunks.'''

    # Check if trained index already exists.
    if not os.path.isfile(get_empty_index_path(env)):

        # Embed training chunks.
        embed_db(env)

        # Train index on embeddings.
        train_on_embeddings(env)

    # Wait for (single-process) training to complete.
    torch.distributed.barrier()

    # Remove embeddings.
    if env.config.retro_index_delete_training_embeddings:
        remove_embeddings(env)


##################################################
# Add to index.
##################################################


def add_to_index():
    '''Add DB chunks to index.'''

    # >>>
    # args = get_retro_args()
    # <<<

    # Get index.
    index = IndexFactory.get_index(env.config.retro_index_type)

    # Get text dataset.
    gpt_dataset = get_merged_train_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    # Add to index.
    output_index_path = index.add(text_dataset)

    return output_index_path


##################################################
# Build index (train + add).
##################################################


def build_index():
    '''Build index.

    Building index involves sequentially running stages above:
    - Train index (on sampled training chunks).
    - Add to index (on all training chunks).
    '''

    # Train index.
    train_index()

    # Add to index.
    add_to_index()
