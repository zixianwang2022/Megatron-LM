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

import glob
import h5py
import os
import torch

# >>>
from lutil import pax
# <<<

def is_valid_path(path):

    f = h5py.File(path, "r")
    # pax({
    #     "path" : path,
    #     "keys" : list(f.keys()),
    # })
    try:
        f["feat"].shape
        return True
    except:
        return False
    finally:
        f.close()

def copy_corpus_dirty_data(args, timer):

    if torch.distributed.get_rank() != 0:
        return

    # Get data paths.
    paths = glob.glob(os.path.join("/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining", "*.hdf5"))
    paths = [ p for p in paths if is_valid_path(p) ]
    paths.sort()
    
    for path_index, src_path in enumerate(paths):

        dst_path = os.path.join("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/data/corpus-dirty", "%04d.hdf5" % path_index)

        # >>>
        # pax({"src_path": src_path, "dst_path": dst_path})
        # <<<

        if os.path.isfile(dst_path):
            continue

        print("copy dirty batch %d / %d." % (path_index, len(paths)))

        f0 = h5py.File(src_path, "r")
        f1 = h5py.File(dst_path, "w")
        f1.create_dataset("data", data = f0["feat"])
        f0.close()
        f1.close()

    # >>>
    pax({"paths": paths})
    # <<<
