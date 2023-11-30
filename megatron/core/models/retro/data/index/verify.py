# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
# import os
# import shutil
import torch
from torch.utils.data import Subset
# from tqdm import tqdm

from megatron.core.models.retro.data.config import RetroPreprocessingConfig
# from megatron.core.models.retro.data.db.utils import (
#     get_merged_sampled_dataset,
#     get_merged_train_dataset,
# )
from megatron.core.models.retro.data.external_libs import h5py
from megatron.core.models.retro.data.utils import (
    GPTToTextDataset,
    get_sampled_blocks_by_rank,
    print_rank_0,
)

from .build import (
    get_text_dataset_for_adding,
    get_text_dataset_for_training,
)
from .factory import IndexFactory
from .utils import (
    get_added_codes_dir,
    get_training_data_block_dir,
)

# >>>
from lutil import pax, print_seq
# <<<


##################################################
# Verify trained index.
##################################################


def verify_training_embeddings(config: RetroPreprocessingConfig) -> None:
    '''Verify training embeddings.

    Steps:
    - Randomly sample subset of text dataset blocks.
    - Embed each block.
    - Compare against saved embeddings.
    '''

    # Training text dataset.
    text_dataset = get_text_dataset_for_training(config)

    # Sample existing blocks.
    blocks = get_sampled_blocks_by_rank(
        dirname=get_training_data_block_dir(config),
        n_samples=len(text_dataset),
        block_size=config.retro_block_size,
        validate=None,
        fraction=config.retro_task_verify,
    )

    assert blocks.n_missing_world == 0

    # Embed & verify blocks.
    embedder = config.retro_bert_embedders.mem
    for block_idx, block in enumerate(blocks.existing):

        # Missing block lists are extended with None to have equal-length
        # lists. Skip the Nones.
        if block is not None:

            # Progress. (*note*: move world progress to here.)
            print_rank_0("embed training block %d / %d ... %s." % (
                block_idx,
                len(blocks.existing),
                block["path"],
            ))

            # Load existing block embeddings.
            with h5py.File(block["path"]) as f:
                existing_embeddings = np.copy(f["data"])

            # Embed block.
            sub_dataset = Subset(text_dataset, range(*block["range"]))
            embeddings = embedder.embed_text_dataset(sub_dataset)

            # Check equality.
            assert np.array_equal(existing_embeddings, embeddings)

            # >>>
            # pax("existing_embeddings, embeddings")
            # <<<

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()

    print_rank_0(" > finished verifying training embeddings.")


##################################################
# Verify filled index.
##################################################


def verify_added_encodings(config):
    '''Verify added encodings.

    Steps:
    - Randomly sample subset of text dataset blocks.
    - Encode each block.
    - Compare against saved encodings.
    '''

    # Index.
    index = IndexFactory.get_index(config.retro_index_type)
    inner_index = index.get_empty_index(config)

    # Text dataset.
    text_dataset = get_text_dataset_for_adding(config)

    # Sample existing blocks.
    def validate(f):
        assert len(f["data"].shape) == 2
    blocks = get_sampled_blocks_by_rank(
        dirname=get_added_codes_dir(config),
        n_samples=len(text_dataset),
        block_size=config.retro_block_size,
        validate=validate,
        fraction=config.retro_task_verify,
    )

    assert blocks.n_missing_world == 0

    # Encode and verify blocks.
    embedder = config.retro_bert_embedders.mem
    for block_idx, block in enumerate(blocks.existing):

        if block is not None:

            # Progress.
            print_rank_0("encode block %d / %d ... %s." % (
                block_idx,
                len(blocks.existing),
                block["path"],
            ))

            # Load existing codes.
            with h5py.File(block["path"]) as f:
                existing_codes = np.copy(f["data"])

            # Encode block.
            codes = index.encode_block(inner_index, embedder, text_dataset, block)

            # Check equality.
            assert np.array_equal(existing_codes, codes)

            # >>>
            pax("existing_codes, codes")
            # <<<

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()

    print_rank_0(" > finished verifying added encodings.")


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
