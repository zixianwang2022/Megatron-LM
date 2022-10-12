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

# from tools.retrieval.data import load_data, save_data
from tools.retrieval.index import Index
from tools.retrieval.utils import print_rank

class OPQIndex(Index):

    def _train(
            self,
            input_data_paths,
            dir_path,
            timer,
    ):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)

        if os.path.isfile(empty_index_path):
            return

        timer.push("load-data")
        inp = load_data(input_data_paths, timer)["data"]
        timer.pop()

        timer.push("init")
        # pq = faiss.IndexPQ(d, self.m, 8)
        # opq = faiss.OPQMatrix(d = d, M = self.m, d2 = self._dout)
        # opq = faiss.OPQMatrix(d = d, M = self.m)
        # opq = faiss.PCAMatrix(d_in = d, d_out = self._dout)
        # opq = faiss.index_factory(d, stage_str)
        # opq = faiss.IndexPreTransform(
        #     faiss.OPQMatrix(d = self.din(), M = self.m, d2 = self.dout()),
        #     faiss.IndexFlatL2(self._dout),
        # )
        din = self.args.nfeats
        dout = self.args.ivf_dim
        opq = faiss.IndexPreTransform(
            faiss.OPQMatrix(
                d = self.args.nfeats,
                M = self.args.pq_m,
                d2 = self.args.ivf_dim,
            ),
            faiss.IndexFlatL2(self.args.ivf_dim),
        )
        self.c_verbose(opq, True)
        timer.pop()

        timer.push("train")
        opq.train(inp)
        timer.pop()

        timer.push("save")
        faiss.write_index(opq, empty_index_path)
        timer.pop()

    def _forward(
            self,
            input_data_paths,
            dir_path,
            timer,
            task,
    ):

        all_output_data_paths, missing_output_data_path_map = \
            self.get_missing_output_data_path_map(
                input_data_paths,
                dir_path,
                task,
            )

        if not missing_output_data_path_map:
            return all_output_data_paths

        timer.push("init")
        index = faiss.read_index(self.args.index_empty_path)
        assert index.chain.size() == 1, "more than opq transform?"
        self.c_verbose(index, True)
        timer.pop()

        timer.push("forward-batches")
        for output_index, (input_index, output_data_path) \
            in enumerate(missing_output_data_path_map.items()):

            timer.push("load-data")
            input_data_path = input_data_paths[input_index]
            inp = load_data([ input_data_path ], timer)["data"]
            timer.pop()

            timer.push("forward-batch")
            print_rank("foward batch %d / %d [ %d vecs ]." % (
                output_index,
                len(missing_output_data_path_map),
                len(inp),
            )) # , flush = True)

            timer.push("forward")
            # out = self.opq.apply_chain(ntrain, inp)
            # out = opq.chain.at(0).apply(inp)
            out = index.chain.at(0).apply(inp)
            timer.pop()

            timer.push("save-data")
            save_data({"data": out}, output_data_path)
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

    def add(self, *args):

        timer = args[-1]

        torch.distributed.barrier()

        timer.push("forward")
        output_data_paths = self._forward(*args, "add")
        timer.pop()

        torch.distributed.barrier()

        return output_data_paths
