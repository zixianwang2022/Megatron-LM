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

from tools.retrieval.utils import mkdir

# >>>
from lutil import pax
# <<<


def get_index_str(args):
    """Faiss notation for index structure."""
    return "OPQ%d_%d,IVF%d_HNSW%d,PQ%d" % (
        args.pq_m,
        args.ivf_dim,
        args.ncluster,
        args.hnsw_m,
        args.pq_m,
    )


# def get_index_dir_path(args):
def get_index_workdir(args):
    """Create sub-directory for this index."""
    
    index_str = get_index_str(args)
    # index_dir_path = os.path.join(
    #     # args.base_dir,
    #     # "index",
    #     args.index_dir,
    #     "%s-%s" % (args.index_ty, args.data_ty),
    #     "%s__t%d" % (index_str, args.ntrain),
    # )
    index_dir_path = os.path.join(
        args.retrieval_workdir,
        "index",
        args.index_ty,
        index_str,
    )

    # pax(0, {
    #     "args" : args,
    #     "index_str" : index_str,
    #     "index_dir_path" : index_dir_path,
    # })

    os.makedirs(index_dir_path, exist_ok = True)

    return index_dir_path
