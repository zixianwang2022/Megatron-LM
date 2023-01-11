# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from megatron import get_retro_args
from tools.retro.db.utils import get_merged_train_dataset
from tools.retro.index.factory import IndexFactory
from tools.retro.utils import GPTToTextDataset


def add_to_index():
    '''Add DB chunks to index.'''

    args = get_retro_args()

    # Get index.
    index = IndexFactory.get_index(args.retro_index_ty)

    # Get text dataset.
    gpt_dataset = get_merged_train_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    # Add to index.
    output_index_path = index.add(text_dataset)

    return output_index_path
