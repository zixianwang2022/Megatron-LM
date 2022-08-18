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

from retrieval.acc.test_index_acc import vis_acc
from retrieval.index.faiss_mono import FaissMonoIndex

# def run_query_acc_pipeline(args, timer):
def plot_query_acc(args, timer):

    if torch.distributed.get_rank() == 0:

        timer.push("get-index-paths")
        base_index = FaissMonoIndex(args)
        test_index = IndexFactory.get_index(args)
        base_index_path = base_index.get_added_index_path(
            args.train_paths,
            args.index_dir_path,
        )
        test_index_path = test_index.get_added_index_path(
            args.train_paths,
            args.index_dir_path,
        )
        index_paths = [
            base_index_path,
            test_index_path,
        ]
        timer.pop()

        pax({
            "base_index" : base_index,
            "test_index" : test_index,
            "base_index_path" : base_index_path,
            "test_index_path" : test_index_path,
            "index_paths" : index_paths,
        })

        timer.push("vis-acc")
        nnbrs = [ 1, 2, 5, 10 ]
        vis_acc(index_paths, nnbrs)
        timer.pop()

    torch.distributed.barrier()
