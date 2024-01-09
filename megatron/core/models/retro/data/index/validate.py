# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch
from torch.utils.data import Subset

from megatron.core.models.retro.data.config import RetroPreprocessingConfig
from megatron.core.models.retro.data.external_libs import h5py
from megatron.core.models.retro.data.utils import (
    GPTToTextDataset,
    get_blocks_by_rank,
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


##################################################
# Validate trained index.
##################################################


def validate_training_embeddings(config: RetroPreprocessingConfig) -> None:
    '''Validate training embeddings.

    Steps:
    - Randomly sample subset of text dataset blocks.
    - Embed each block.
    - Compare against saved embeddings.
    '''

    # Training text dataset.
    text_dataset = get_text_dataset_for_training(config)

    # Sample existing blocks.
    blocks = get_blocks_by_rank(
        dirname=get_training_data_block_dir(config),
        n_samples=len(text_dataset),
        block_size=config.retro_block_size,
        validate=None,
        sample=config.retro_task_validate,
    )

    assert blocks.n_missing_world == 0

    # Embed & validate blocks.
    embedder = config.retro_bert_embedders.mem
    for block_idx, block in enumerate(blocks.existing):

        # Missing block lists are extended with None to have equal-length
        # lists. Skip the Nones.
        if block is not None:

            # Progress. (*note*: move world progress to here.)
            print_rank_0(
                "embed training block %d / %d ... %s."
                % (block_idx, len(blocks.existing), block["path"],)
            )

            # Load existing block embeddings.
            with h5py.File(block["path"]) as f:
                existing_embeddings = np.copy(f["data"])

            # Embed block.
            sub_dataset = Subset(text_dataset, range(*block["range"]))
            embeddings = embedder.embed_text_dataset(sub_dataset, "train")

            # >>>
            # # embeddings_1 = embedder.embed_text_dataset(sub_dataset, "train-1")

            # import math
            # # import torch.nn.functional as F
            # from lutil import pax
            # # err = math.sqrt(F.mse_loss(embeddings, existing_embeddings).item())
            # err = math.sqrt(((embeddings - existing_embeddings)**2).mean().item())
            # print("~~~~~~~~~~~ err: %f ~~~~~~~~~~~" % err)
            # # pax("block, existing_embeddings, embeddings, err")
            # <<<

            # Check equality.
            # >>>
            print_rank_0(" > validate.")
            assert np.array_equal(existing_embeddings, embeddings)
            # <<<

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()

    print_rank_0(" > finished validating training embeddings.")


##################################################
# Validate filled index.
##################################################


def validate_added_encodings(config):
    '''Validate added encodings.

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

    blocks = get_blocks_by_rank(
        dirname=get_added_codes_dir(config),
        n_samples=len(text_dataset),
        block_size=config.retro_block_size,
        validate=validate,
        sample=config.retro_task_validate,
    )

    assert blocks.n_missing_world == 0

    # Encode and validate blocks.
    embedder = config.retro_bert_embedders.mem
    for block_idx, block in enumerate(blocks.existing):

        if block is not None:

            # Progress.
            print_rank_0(
                "encode block %d / %d ... %s." % (block_idx, len(blocks.existing), block["path"],)
            )

            # Load existing codes.
            with h5py.File(block["path"]) as f:
                existing_codes = np.copy(f["data"])

            # Encode block.
            embeddings, codes = index.encode_block(inner_index, embedder, text_dataset, block)

            # >>>
            # import math
            # import os
            # n = (existing_codes != codes).sum() / existing_codes.size
            # e = math.sqrt(((existing_codes - codes)**2).mean().item())
            # print("~~~~~~~~~~~ n %f, e %f ... '%s' ~~~~~~~~~~~" % (n, e, os.path.basename(block["path"])))

            # from lutil import pax
            # pax("existing_codes, codes")
            # <<<

            # Check equality.
            # >>>
            print_rank_0(" > validate.")
            assert np.array_equal(existing_codes, codes)
            # <<<

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()

    print_rank_0(" > finished validating added encodings.")


##################################################
# Validate index (trained + filled).
##################################################


def validate_index(config):
    '''Validate index.

    Validating index involves sequentially running stages above:
    - Validate trained index.
    - Validate filled index.
    '''

    # Validate training embeddings.
    validate_training_embeddings(config)

    # Validate added codes.
    validate_added_codes(config)
