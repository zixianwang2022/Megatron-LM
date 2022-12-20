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

import os

from megatron import get_retro_args
from tools.bert_embedding.utils import get_missing_blocks
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
    missing_blocks = get_missing_blocks(new_dir, n_samples, block_size)

    pax({
        "old_dir" : old_dir,
        "new_dir" : new_dir,
        "gpt_dataset" : gpt_dataset,
        "n_samples" : n_samples,
        "block_size" : block_size,
        "missing_blocks" : missing_blocks,
    })
