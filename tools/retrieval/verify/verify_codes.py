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
import numpy as np
import torch

from tools.retrieval.add import add_to_index
from tools.retrieval.index import FaissBaseIndex

# >>>
from lutil import pax
# <<<

def verify_codes(args, timer):

    timer.push("add-base")
    base_index = FaissBaseIndex(args)
    base_index_path = base_index.add(args.add_paths, args.index_dir_path, timer)
    timer.pop()

    timer.push("add-test")
    test_index_path = add_to_index(args, timer)
    timer.pop()

    torch.distributed.barrier()

    if torch.distributed.get_rank() != 0:
        return

    # timer.push("get-index-paths")
    # base_index = FaissMonoIndex(args)
    # test_index = IndexFactory.get_index(args)
    # base_index_path = base_index.get_added_index_path(
    #     args.add_paths,
    #     args.index_dir_path,
    # )
    # test_index_path = test_index.get_added_index_path(
    #     args.add_paths,
    #     args.index_dir_path,
    # )
    # timer.pop()

    # >>>
    # pax({
    #     "base_index_path" : base_index_path,
    #     "test_index_path" : test_index_path,
    # })
    # <<<

    # Read indexes.
    timer.push("read")
    base_index = faiss.read_index(base_index_path)
    test_index = faiss.read_index(test_index_path)
    base_invlists = faiss.extract_index_ivf(base_index).invlists
    test_invlists = faiss.extract_index_ivf(test_index).invlists
    timer.pop()

    # # ###########################
    # # Test #1: Serialize indexes.
    # # ###########################

    # base_np = faiss.serialize_index(base_index)
    # test_np = faiss.serialize_index(test_index)
    # # base_invlist_np = faiss.serialize_index(base_invlists)
    # # test_invlist_np = faiss.serialize_index(test_invlists)

    # pax({
    #     "base_np" : str(base_np),
    #     "test_np" : str(test_np),
    #     "base_np / hash" : hash(base_np.tobytes()),
    #     "test_np / hash" : hash(test_np.tobytes()),
    #     # "base_invlist_np" : str(base_invlist_np),
    #     # "test_invlist_np" : str(test_invlist_np),
    #     # "base_invlist_np / hash" : hash(base_invlist_np.tobytes()),
    #     # "test_invlist_np / hash" : hash(test_invlist_np.tobytes()),
    #     "equal" : np.array_equal(base_np, test_np),
    # })

    ############################################
    # Test #2: Compare each list's ids/codes.
    # - Note: only relevant if above test fails.
    ############################################

    # Verify same list size.
    assert base_index.ntotal == test_index.ntotal

    # Compare each list's ids/codes.
    timer.push("add")
    for list_id in range(args.ncluster):

        if list_id % 100000 == 0:
            print("verify list %d / %d." % (list_id, args.ncluster))

        # Get list size, ids, codes.
        base_list_size = base_invlists.list_size(list_id)
        base_ids = np.empty((base_list_size,), dtype = "i8")
        base_codes = np.empty((base_list_size, args.pq_m), dtype = "uint8")
        test_list_size = test_invlists.list_size(list_id)
        test_ids = np.empty((test_list_size,), dtype = "i8")
        test_codes = np.empty((test_list_size, args.pq_m), dtype = "uint8")

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

        # Verify list size, ids, codes.
        assert base_list_size == test_list_size
        assert np.array_equal(base_ids, test_ids)
        assert np.array_equal(base_codes, test_codes)

        # pax({
        #     "base_list_size" : base_list_size,
        #     "test_list_size" : test_list_size,
        #     "base_ids" : base_ids,
        #     "test_ids" : test_ids,
        #     "base_codes" : base_codes,
        #     "test_codes" : test_codes,
        #     "ids / equal" : np.array_equal(base_ids, test_ids),
        #     "codes / equal" : np.array_equal(base_codes, test_codes),
        # })

    timer.pop()

    # pax({
    #     "base_invlists" : base_invlists,
    #     "test_invlists" : test_invlists,
    # })

    # Final sync. [ unnecessary ]
    # torch.distributed.barrier()

    # print_seq([ base_index_path, test_index_path ])
