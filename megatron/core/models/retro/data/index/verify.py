# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# import numpy as np
# import os
# import shutil
# import torch
# from tqdm import tqdm

# from megatron.core.models.retro.data.db.utils import (
#     get_merged_sampled_dataset,
#     get_merged_train_dataset,
# )
# from megatron.core.models.retro.data.external_libs import h5py
# from megatron.core.models.retro.data.utils import GPTToTextDataset

# from .factory import IndexFactory
# from .utils import (
#     get_training_data_block_dir,
#     get_training_data_block_paths,
#     get_training_data_merged_path,
#     get_training_data_root_dir,
# )


##################################################
# Verify trained index.
##################################################


# def get_empty_index_path(config):
#     '''Path of empty index.'''
#     index = IndexFactory.get_index(config.retro_index_type)
#     empty_index_path = index.get_empty_index_path(config)
#     return empty_index_path


# def get_block_nload(block_path, load_fraction):
#     with h5py.File(block_path) as fi:
#         return int(load_fraction * fi["data"].shape[0])


# def merge_embedding_blocks(config):

#     if torch.distributed.get_rank() != 0:
#         return

#     # Get block, merged paths.
#     load_fraction = config.retro_index_train_load_fraction
#     block_paths = get_training_data_block_paths(config)
#     bin_path = get_training_data_merged_path(config)

#     # Skip, if already built.
#     if os.path.exists(bin_path):
#         return

#     # Merge blocks.
#     with open(bin_path, "wb") as fo:
#         byte_offset = 0
#         for block_idx, block_path in \
#             enumerate(tqdm(block_paths, "merge train embeddings")):
#             with h5py.File(block_path) as fi:

#                 nload = get_block_nload(block_path, load_fraction)
#                 block = np.array(fi["data"][:nload], copy = False)

#                 fo.write(block.tobytes())

#                 byte_offset += block.size * block.itemsize
#                 fo.seek(byte_offset)


def verify_trained_index(config):
    '''Verify trained index.

    Steps:
    - Randomly sample subset of chunk blocks.
    - Embed each block
    - Compare against saved embeddings.
    '''

    blocks = get_sampled_blocks_by_rank(
        project_dir = config.retro_project_dir,
        n_samples: int,
        block_size: int,
        validate: Callable = None,
    )

    pax("blocks")

    merged_train_data_path = get_training_data_merged_path(config)
    if os.path.exists(merged_train_data_path):
        return

    # Get db dataset.
    gpt_dataset = get_merged_sampled_dataset(
        project_dir=config.retro_project_dir,
        chunk_length=config.retro_gpt_chunk_length,
        eod_token_id=config.retro_tokenizers.gpt.eod,
    )

    text_dataset = GPTToTextDataset(gpt_dataset, config.retro_tokenizers.gpt)

    # Embed dataset.
    # embedder = config.retro_bert_embedders.disk
    embedder.embed_text_dataset("index",
                                get_training_data_block_dir(config),
                                text_dataset)

    # Merge embeddings.
    merge_embedding_blocks(config)


# def train_on_embeddings(config):
#     '''Train index on embedded DB chunks.'''
#     index = IndexFactory.get_index(config.retro_index_type)
#     index.train(config)


# def remove_embeddings(config):
#     '''Remove embeddings after training.'''
#     torch.distributed.barrier()
#     if torch.distributed.get_rank() != 0:
#         return
#     empty_index_path = get_empty_index_path(config)
#     assert os.path.isfile(empty_index_path)
#     shutil.rmtree(get_training_data_root_dir(config), ignore_errors=True)


def verify_trained_index(config):
    '''Verify trained index.'''

    # Check if trained index already exists.
    if not os.path.isfile(get_empty_index_path(config)):

        # Embed training chunks.
        embed_db(config)

        # Train index on embeddings.
        train_on_embeddings(config)

    # Wait for (single-process) training to complete.
    torch.distributed.barrier()

    # Remove embeddings.
    if config.retro_index_delete_training_embeddings:
        remove_embeddings(config)


##################################################
# Verify filled index.
##################################################


def verify_filled_index(config):
    '''Verify filled index.'''

    # Get index.
    index = IndexFactory.get_index(config.retro_index_type)

    # Get text dataset.
    gpt_dataset = get_merged_train_dataset(
        project_dir=config.retro_project_dir,
        chunk_length=config.retro_gpt_chunk_length,
        eod_token_id=config.retro_tokenizers.gpt.eod,
    )
    text_dataset = GPTToTextDataset(gpt_dataset, config.retro_tokenizers.gpt)

    # Fill index.
    output_index_path = index.add(config, text_dataset)

    return output_index_path


##################################################
# Verify index (trained + filled).
##################################################


def verify_index(config):
    '''Verify index.

    Verifying index involves sequentially running stages above:
    - Verify trained index.
    - Verify filled index.
    '''

    # Verify trained index.
    verify_trained_index(config)

    # Verify filled index.
    verify_filled_index(config)
