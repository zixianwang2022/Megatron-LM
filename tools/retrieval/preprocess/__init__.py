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
import torch

# from .id import save_document_ids
# from .offset import save_document_offsets
from .index_chunks import build_chunk_indexes

def preprocess_chunks(args, timer):

    if torch.distributed.get_rank() != 0:
        return

    # Preprocessing workdir.
    workdir = os.path.join(args.retrieval_workdir, "preprocess")
    os.makedirs(workdir, exist_ok = True)

    # Stages.
    build_chunk_indexes(args, workdir)
    save_document_offsets(args, workdir)
    save_document_ids(args, workdir, timer)
