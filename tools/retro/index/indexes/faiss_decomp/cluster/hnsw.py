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
import os
import torch

# from tools.retro.data import load_data, save_data
from tools.retro.index import Index
from tools.retro.utils import print_rank

class HNSWIndex(Index):

    def _train(
            self,
            input_data_paths,
            centroid_data_paths,
            dir_path,
            timer,
    ):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)

        if os.path.isfile(empty_index_path):
            return

        timer.push("load-data")
        centroids = load_data(centroid_data_paths, timer)["centroids"]
        timer.pop()

        timer.push("init")
        hnsw = faiss.IndexHNSWFlat(self.args.ivf_dim, self.args.hnsw_m)
        self.c_verbose(hnsw, True)
        timer.pop()

        timer.push("train")
        hnsw.train(centroids)
        timer.pop()

        timer.push("add")
        hnsw.add(centroids)
        timer.pop()

        timer.push("save")
        faiss.write_index(hnsw, empty_index_path)
        timer.pop()

    def _forward(
            self,
            input_data_paths,
            _, # centroid_data_paths,
            dir_path,
            timer,
            task,
    ):

        empty_index_path = self.get_empty_index_path(dir_path)

        all_output_data_paths, missing_output_data_path_map = \
            self.get_missing_output_data_path_map(input_data_paths,dir_path,task)

        all_output_data_paths = [ {
            "data" : i,
            "centroid_ids" : o,
        } for i, o in zip(input_data_paths, all_output_data_paths) ]

        if not missing_output_data_path_map:
            return all_output_data_paths

        timer.push("init")
        hnsw = faiss.read_index(empty_index_path)
        self.c_verbose(hnsw, True)
        timer.pop()

        timer.push("forward-batches")
        for output_index, (input_index, output_data_path) in \
            enumerate(missing_output_data_path_map.items()):

            timer.push("load-data")
            input_data_path = input_data_paths[input_index]
            inp = load_data([ input_data_path ], timer)["data"]
            timer.pop()

            print_rank("foward batch %d / %d. [ %d vecs ]" % (
                output_index,
                len(missing_output_data_path_map),
                len(inp),
            )) # , flush = True)

            timer.push("forward-batch")

            timer.push("search")
            dists, centroid_ids = hnsw.search(inp, 1)
            timer.pop()

            timer.push("save-data")
            save_data({
                "centroid_ids" : centroid_ids,
            }, output_data_path)
            timer.pop()

            timer.pop()

        timer.pop()

        return all_output_data_paths

    def train(self, *args):

        raise Exception("train mono instead.")

        timer = args[-1]

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            timer.push("train")
            self._train(*args)
            timer.pop()

        torch.distributed.barrier()

        timer.push("forward")
        output_data_paths = self._forward(*args, "train")
        timer.pop()

        torch.distributed.barrier()

        return output_data_paths

    def add(
            self,
            input_data_paths,
            dir_path,
            timer,
    ):

        torch.distributed.barrier()

        timer.push("forward")
        output_data_paths = self._forward(
            input_data_paths,
            None,
            dir_path,
            timer,
            "add",
        )
        timer.pop()

        torch.distributed.barrier()

        return output_data_paths
