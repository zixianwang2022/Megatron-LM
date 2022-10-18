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

from datetime import timedelta
import faiss
import os
import torch

from tools.retro.embed.utils import load_data
from tools.retro.index import Index
from tools.retro.index.utils import get_index_str
from tools.retro.utils import print_rank

# >>>
from lutil import pax
# <<<


class FaissBaseIndex(Index):

    def _train(self, input_data_paths, dir_path, timer):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)

        # Index already exists? -> return.
        if os.path.isfile(empty_index_path):
            return

        # Load data.
        timer.push("load-data")
        inp = load_data(input_data_paths, timer)["data"]
        timer.pop()

        # Init index.
        timer.push("init")
        index_str = get_index_str(self.args)
        index = faiss.index_factory(self.args.nfeats, index_str)
        timer.pop()

        # Move to GPU.
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = \
            faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index
        self.c_verbose(index, True)
        self.c_verbose(index_ivf, True)
        self.c_verbose(index_ivf.quantizer, True)
        self.c_verbose(index_ivf.clustering_index, True)

        # Train index.
        timer.push("train")
        index.train(inp)
        timer.pop()

        # Save index.
        timer.push("save")
        faiss.write_index(index, empty_index_path)
        timer.pop()


    # def train(self, input_data_paths, dir_path, timer):
    # def train(self, workdir, timer):
    def train(self, input_data_paths, dir_path, timer):

        if torch.distributed.get_rank() == 0:
            timer.push("train")
            self._train(input_data_paths, dir_path, timer)
            timer.pop()

        torch.distributed.barrier()


    def _add(self, input_data_paths, dir_path, timer):

        # Set num threads (torch.distributed reset it to 1).
        faiss.omp_set_num_threads(64)

        # pax(0, {"nthreads": faiss.omp_get_max_threads()})

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)
        added_index_path = self.get_added_index_path(input_data_paths, dir_path)

        # pax(0, {
        #     "empty_index_path" : empty_index_path,
        #     "added_index_path" : added_index_path,
        # })

        if os.path.isfile(added_index_path):
            return

        timer.push("init")
        index = faiss.read_index(empty_index_path)
        timer.pop()

        for batch_id, input_data_path in enumerate(input_data_paths):

            print_rank("faiss-base / add ... batch %d / %d." % (
                batch_id,
                len(input_data_paths),
            ))

            timer.push("load-data")
            inp = load_data([ input_data_path ], timer)["data"]
            timer.pop()

            # >>>
            index_ivf = faiss.extract_index_ivf(index)
            # clustering_index = \
            #     faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
            # index_ivf.clustering_index = clustering_index
            self.c_verbose(index, True)
            self.c_verbose(index_ivf, True)
            self.c_verbose(index_ivf.quantizer, True)
            # self.c_verbose(index_ivf.clustering_index, True)
            # <<<

            timer.push("add")
            index.add(inp)
            timer.pop()

        timer.push("save")
        faiss.write_index(index, added_index_path)
        timer.pop()


    def add(self, input_data_paths, dir_path, timer):

        if torch.distributed.get_rank() == 0:
            timer.push("add")
            self._add(input_data_paths, dir_path, timer)
            timer.pop()

        torch.distributed.barrier()

        return self.get_added_index_path(input_data_paths, dir_path)

