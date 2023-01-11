# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os

from megatron import get_retro_args
from tools.bert_embedding.utils import get_missing_blocks, get_index_path_map
from tools.retro.db.utils import get_merged_sampled_dataset
from tools.retro.index.utils import get_index_dir, get_training_data_dir

# >>>
from lutil import pax
# <<<


def update_training_block_size():

    args = get_retro_args()

    old_dir = get_training_data_dir()
    new_dir = os.path.join(get_index_dir(), "training_data_tmp_NEW")
    os.makedirs(new_dir, exist_ok = True)

    gpt_dataset = get_merged_sampled_dataset()
    n_samples = len(gpt_dataset)
    block_size = args.retro_block_size
    new_missing_blocks = get_missing_blocks(new_dir, n_samples, block_size)

    old_idx_path_map = get_index_path_map(old_dir)

    pax({
        "old_dir" : old_dir,
        "new_dir" : new_dir,
        "gpt_dataset" : gpt_dataset,
        "n_samples" : n_samples,
        "block_size" : block_size,
        "missing_blocks" : missing_blocks,
        "old_idx_path_map" : old_idx_path_map,
    })
