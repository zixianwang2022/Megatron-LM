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
import h5py
import numpy as np
import os
import torch
from tqdm import tqdm

# from tools.retro.add import add_to_index
# from tools.retro.index import FaissBaseIndex, IndexFactory

from ..utils import get_training_data_block_paths

# >>>
from lutil import pax
# <<<


def verify_codes(timer):

    # timer.push("add-base")
    # base_index = FaissBaseIndex(args)
    # base_index_path = base_index.add(args.add_paths, args.index_dir_path, timer)
    # timer.pop()

    # timer.push("add-test")
    # test_index = IndexFactory.get_index(args)
    # test_index_path = test_index.add(args.add_paths, args.index_dir_path, timer)
    # # test_index_path = add_to_index(args, timer)
    # timer.pop()

    # torch.distributed.barrier()

    if torch.distributed.get_rank() != 0:
        return

    # >>>
    empty_index_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/empty.faissindex"
    base_index_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/added_base.faissindex"
    test_index_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/added_0667_0000-0666.faissindex"

    if not os.path.exists(base_index_path):
        base_index = faiss.read_index(empty_index_path)
        input_paths = get_training_data_block_paths()
        for input_index, input_path in enumerate(tqdm(input_paths)):
            # >>>
            # if input_index == 10:
            #     break
            # <<<
            with h5py.File(input_path) as f:
                base_index.add(np.copy(f["data"]))
        # pax({
        #     "input_paths" : input_paths,
        #     "base_index" : base_index,
        # })
        faiss.write_index(base_index, base_index_path)
        raise Exception("added base.")


    # test_index = faiss.read_index(

    # pax({
    #     "base_index" : base_index,
    #     "test_index" : test_index,
    # })
    # <<<

    # Read indexes.
    timer.push("read")
    # base_index = faiss.read_index(base_index_path)
    # test_index = faiss.read_index(test_index_path)
    base_index = faiss.read_index(base_index_path, faiss.IO_FLAG_MMAP)
    test_index = faiss.read_index(test_index_path, faiss.IO_FLAG_MMAP)
    base_invlists = faiss.extract_index_ivf(base_index).invlists
    test_invlists = faiss.extract_index_ivf(test_index).invlists
    timer.pop()

    # pax({
    #     "base ivf" : faiss.extract_index_ivf(base_index),
    #     "base ils" : base_invlists,
    # })

    # # ###########################
    # # Test #1: Serialize indexes.
    # # ###########################

    # base_np = faiss.serialize_index(base_index)
    # test_np = faiss.serialize_index(test_index)
    # # base_invlist_np = faiss.serialize_index(base_invlists)
    # # test_invlist_np = faiss.serialize_index(test_invlists)

    ############################################
    # Test #2: Compare each list's ids/codes.
    # - Note: only relevant if above test fails.
    ############################################

    # Verify same list size.
    assert base_index.ntotal == test_index.ntotal

    # Compare each list's ids/codes.
    timer.push("add")
    nlist = base_invlists.nlist
    code_size = base_invlists.code_size
    # for list_id in range(args.ncluster):
    for list_id in tqdm(range(nlist)):

        # if list_id % 100000 == 0:
        #     print("verify list %d / %d." % (list_id, nlist), flush = True)

        # Get list size, ids, codes.
        base_list_size = base_invlists.list_size(list_id)
        base_ids = np.empty((base_list_size,), dtype = "i8")
        base_codes = np.empty((base_list_size, code_size), dtype = "uint8")
        test_list_size = test_invlists.list_size(list_id)
        test_ids = np.empty((test_list_size,), dtype = "i8")
        test_codes = np.empty((test_list_size, code_size), dtype = "uint8")

        base_id_ptr = base_invlists.get_ids(list_id)
        base_code_ptr = base_invlists.get_codes(list_id)
        test_id_ptr = test_invlists.get_ids(list_id)
        test_code_ptr = test_invlists.get_codes(list_id)
        faiss.memcpy(faiss.swig_ptr(base_ids), base_id_ptr, base_ids.nbytes)
        faiss.memcpy(faiss.swig_ptr(base_codes), base_code_ptr, base_codes.nbytes)
        faiss.memcpy(faiss.swig_ptr(test_ids), test_id_ptr, test_ids.nbytes)
        faiss.memcpy(faiss.swig_ptr(test_codes), test_code_ptr, test_codes.nbytes)
        base_invlists.release_ids(list_id, base_id_ptr)
        base_invlists.release_codes(list_id, base_code_ptr)
        test_invlists.release_ids(list_id, test_id_ptr)
        test_invlists.release_codes(list_id, test_code_ptr)

        # pax({
        #     "base_ids" : base_ids,
        #     "base_codes" : base_codes,
        # })

        # Verify list size, ids, codes.
        assert base_list_size == test_list_size
        assert np.array_equal(base_ids, test_ids)
        assert np.array_equal(base_codes, test_codes)

    print("verified %d codes." % base_index.ntotal)

    timer.pop()

    # Final sync. [ unnecessary ]
    # torch.distributed.barrier()
