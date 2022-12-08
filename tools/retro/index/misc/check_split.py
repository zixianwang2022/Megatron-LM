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

import faiss
from tools.retro.db.utils import get_indexed_dataset_infos

# >>>
from lutil import pax
# <<<


def check_index_train_valid_split(timer):

    indexed_dataset_infos = get_indexed_dataset_infos()

    index_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/added_0667_0000-0666.faissindex"
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)

    pax(0, {
        "indexed_dataset_infos" : indexed_dataset_infos,
        "indexed_dataset_infos / 0" : indexed_dataset_infos[0],
        "index_path" : index_path,
        "index" : index,
    })
