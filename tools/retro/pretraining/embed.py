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

from tools.bert_embedding import embed_text_datasets

from .chunk_dataset import get_text_chunk_dataset_map

# >>>
from lutil import pax
# <<<


def embed_pretraining_chunks(args, timer):

    # Data stuff.
    text_dataset_map = get_text_chunk_dataset_map(args)
    
    # Embed.
    embed_text_datasets(text_dataset_map,
                        args.retro_bert_max_chunk_length,
                        args.retro_block_size)
