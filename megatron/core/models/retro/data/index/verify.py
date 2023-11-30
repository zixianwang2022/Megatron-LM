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


def verify_trained_index(config):
    '''Verify trained index.

    Steps:
    - Randomly sample subset of chunk blocks.
    - Embed each block
    - Compare against saved embeddings.
    '''

    blocks = get_sampled_blocks_by_rank(
        dirname=get_training_data_block_dir(config),
        n_samples=len(text_dataset),
        block_size=config.retro_block_size,
        validate=None,
        fraction=config.retro_task_verify,
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


##################################################
# Verify filled index.
##################################################


def verify_filled_index(config):
    '''Verify filled index.'''

    raise Exception("hi.")

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
