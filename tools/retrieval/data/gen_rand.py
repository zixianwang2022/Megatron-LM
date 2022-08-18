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

import h5py
import numpy as np
import os
import torch

from tools.retrieval import utils

def gen_rand_data(args, timer):

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # existing_
    nvecs = int(1e9)
    for key, batch_size in [
            ("1m", int(1e6)),
            # ("100k", int(1e5)),
    ]:

        # base_path = os.path.join(args.base_dir, "rand-%s" % key)
        # utils.makedir(base_path)
        base_path = utils.mkdir(os.path.join(
            args.base_dir,
            "data",
            "rand-%s" % key,
        ))

        num_batches = int(nvecs / batch_size)
        # for batch_index in range(num_batches): # single process
        # for batch_index in range(0, num_batches, world_size):
        # for batch_index in range(
        #         0 * 1000 + rank, # ... 100k
        #         1 * 1000,
        #         world_size,
        # ):
        group_id = 0
        for batch_index in range(
                group_id * 100 + rank, # ... 1m
                (group_id + 1) * 100, #          + 10,
                world_size,
        ):

            path = os.path.join(base_path, "%s.hdf5" % str(batch_index).zfill(6))

            if os.path.isfile(path):
                try:
                    f = h5py.File(path, "r")
                    shape = f["data"].shape
                    continue
                except:
                    # raise Exception("delete '%s'." % os.path.basename(path))
                    os.remove(path)
                finally:
                    f.close()
                # raise Exception("file exists.")
                # continue

            utils.print_rank("create rand-%s, batch %d / %d." % (
                key,
                batch_index,
                num_batches,
            ))

            data = np.random.rand(batch_size, args.nfeats).astype("f4")

            f = h5py.File(path, "w")
            f.create_dataset("data", data = data)
            f.close()
