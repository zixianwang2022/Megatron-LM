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
import re
import torch

# from tools.retro.data import load_data, save_data
from tools.retro.index import Index
# from tools.retro.utils import print_rank

class IVFPQIndex(Index):

    @classmethod
    def c_cpu_to_gpu(cls, index):
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index.d))
        index.clustering_index = clustering_index

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # train
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        index = faiss.IndexIVFPQ(
            faiss.IndexFlat(self.args.ivf_dim),
            self.args.ivf_dim,
            self.args.ncluster,
            self.args.pq_m,
            self.args.pq_nbits,
        )
        self.c_verbose(index, True)
        self.c_verbose(index.quantizer, True)
        self.c_cpu_to_gpu(index)
        self.c_verbose(index.clustering_index, True)
        timer.pop()

        timer.push("train")
        index.train(inp)
        timer.pop()

        timer.push("save")
        faiss.write_index(index, empty_index_path)
        timer.pop()

    def get_centroid_data_path(self, dir_path):
        return self.get_output_data_path(dir_path, "train", "centroids")

    def _forward_centroids(
            self,
            input_data_paths,
            dir_path,
            timer,
            task,
    ):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)
        output_data_path = self.get_centroid_data_path(dir_path)

        if not os.path.isfile(output_data_path):

            timer.push("init")
            index = faiss.read_index(empty_index_path)
            self.c_verbose(index, True)
            self.c_verbose(index.quantizer, True)
            # self.c_cpu_to_gpu(index) # ... unnecessary for centroid reconstruct
            # self.c_verbose(index.clustering_index, True) # ... only after gpu
            timer.pop()

            timer.push("save-data")
            centroids = index.quantizer.reconstruct_n(0, self.args.ncluster)
            save_data({"centroids": centroids}, output_data_path)
            timer.pop()

        return [ output_data_path ]

    def train(self, input_data_paths, dir_path, timer):

        raise Exception("train mono instead.")

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:

            timer.push("train")
            self._train(input_data_paths, dir_path, timer)
            timer.pop()

            timer.push("forward")
            output_data_paths = self._forward_centroids(
                input_data_paths,
                dir_path,
                timer,
                "train",
            )
            timer.pop()

        torch.distributed.barrier()

        # return output_data_paths
        return [ self.get_centroid_data_path(dir_path) ]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # add
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def get_num_rows(cls, num_batches):
        return int(np.ceil(np.log(num_batches) / np.log(2))) + 1
    @classmethod
    def get_num_cols(cls, num_batches, row):
        world_size = torch.distributed.get_world_size()
        return int(np.ceil(num_batches / world_size / 2**row))

    def get_partial_index_path_map(
            self,
            input_data_paths,
            dir_path,
            row,
            col,
            rank = None,
    ):

        # rank = torch.distributed.get_rank()
        rank = torch.distributed.get_rank() if rank is None else rank
        world_size = torch.distributed.get_world_size()
        num_batches = len(input_data_paths)

        batch_str_len = int(np.ceil(np.log(num_batches) / np.log(10))) + 1
        zf = lambda b : str(b).zfill(batch_str_len)

        # First batch id.
        batch_id_0 = 2**row * (rank + col * world_size)

        if batch_id_0 >= num_batches:
            return None

        # Other batch ids.
        if row == 0:
            batch_id_1 = batch_id_2 = batch_id_3 = batch_id_0
        else:
            batch_full_range = 2**row
            batch_half_range = int(batch_full_range / 2)
            batch_id_1 = batch_id_0 + batch_half_range - 1
            batch_id_2 = batch_id_1 + 1
            batch_id_3 = batch_id_2 + batch_half_range - 1

        # Batch ranges.
        def get_batch_range(b0, b1):
            if b1 >= num_batches:
                b1 = num_batches - 1
            if b0 >= num_batches:
                return None
            return b0, b1

        def batch_range_to_index_path(_row, _range):
            if _range is None:
                return None
            else:
                return os.path.join(
                    dir_path,
                    # "partial_r%d_%s-%s.faissindex" % (
                    #     _row,
                    #     *[zf(b) for b in _range],
                    # ),
                    # "partial_%s_%s-%s.faissindex" % (
                    "added_%s_%s-%s.faissindex" % (
                        # _row, # ... using row id disallows cross-row sharing
                        zf(_range[-1] - _range[0] + 1),
                        *[zf(b) for b in _range],
                    ),
                )

        input_batch_ranges = [
            get_batch_range(batch_id_0, batch_id_1),
            get_batch_range(batch_id_2, batch_id_3),
        ]
        output_batch_range = get_batch_range(batch_id_0, batch_id_3)

        # Path map.
        path_map = {
            "batch_id" : batch_id_0,
            "num_batches" : num_batches,
            "output_index_path" :
            batch_range_to_index_path(row, output_batch_range),
        }
        if row == 0:
            path_map["input_data_path"] = input_data_paths[batch_id_0]
        else:
            input_index_paths = \
                [batch_range_to_index_path(row-1, r) for r in input_batch_ranges]
            input_index_paths = [ p for p in input_index_paths if p is not None ]

            if not input_index_paths:
                return None

            path_map["input_index_paths"] = input_index_paths

        # Return.
        return path_map

    def get_missing_index_paths(
            self,
            input_data_paths,
            dir_path,
            timer,
            num_rows,
    ):

        world_size = torch.distributed.get_world_size()
        num_batches = len(input_data_paths)

        missing_index_paths = set()
        # missing_index_path_grid = []
        for row in range(num_rows - 1, -1, -1):

            # missing_index_path_row = []
            # missing_index_path_grid.append(missing_index_path_row)

            num_cols = self.get_num_cols(num_batches, row)
            for col in range(num_cols):

                for rank in range(world_size):

                    # Input/output index paths.
                    path_map = self.get_partial_index_path_map(
                        input_data_paths,
                        dir_path,
                        row,
                        col,
                        rank,
                    )

                    # Handle edge cases.
                    if path_map is None:
                        continue

                    output_path = path_map["output_index_path"]
                    input_paths = path_map.get("input_index_paths", [])

                    if row == num_rows - 1 and not os.path.isfile(output_path):
                        missing_index_paths.add(output_path)

                    if output_path in missing_index_paths:
                        missing_input_paths = \
                            [ p for p in input_paths if not os.path.isfile(p) ]
                        missing_index_paths.update(missing_input_paths)

        return missing_index_paths

    # def add_partial(self, input_data_paths, dir_path, timer, row, col):
    # def add_partial(self, partial_index_path_map, timer):
    # def init_partial(self, partial_index_path_map, dir_path, timer):
    # def init_partial_index(self, partial_index_path_map, dir_path, timer):
    def init_partial(self, partial_index_path_map, dir_path, timer):

        # >>>
        # row = 2; col = 2
        # row = 3; col = 1
        # <<<

        empty_index_path = self.get_empty_index_path(dir_path)
        partial_index_path = partial_index_path_map["output_index_path"]
        input_data_path_item = partial_index_path_map["input_data_path"]

        if os.path.isfile(partial_index_path):
            return

        timer.push("load-data")
        input_data_path = input_data_path_item["data"]
        input_data = load_data([input_data_path], timer)["data"].astype("f4")
        cluster_id_path = input_data_path_item["centroid_ids"]
        cluster_ids = \
            load_data([cluster_id_path], timer)["centroid_ids"].astype("i8")
        timer.pop()

        nvecs = len(input_data)
        print_rank("ivfpq / add / partial,  batch %d / %d. [ %d vecs ]" % (
            partial_index_path_map["batch_id"],
            partial_index_path_map["num_batches"],
            nvecs,
        ))

        timer.push("read")
        index = faiss.read_index(empty_index_path)
        # self.c_verbose(index, True) # with batch_size 1M ... too fast/verbose
        # self.c_verbose(index.quantizer, True)
        timer.pop()

        timer.push("add")
        index.add_core(
            n = nvecs,
            x = self.swig_ptr(input_data),
            # xids = self.swig_ptr(np.arange(*meta["vec_range"], dtype = "i8")),
            xids = self.swig_ptr(np.arange(nvecs, dtype = "i8")),
            precomputed_idx = self.swig_ptr(cluster_ids),
        )
        timer.pop()

        timer.push("write")
        faiss.write_index(index, partial_index_path)
        timer.pop()

    # def merge_partial_indexes(self, partial_index_path_map, dir_path, timer):
    def merge_partial(self, partial_index_path_map, dir_path, timer):
        '''Merge partial indexes.'''

        # Index paths.
        empty_index_path = self.get_empty_index_path(dir_path)
        output_index_path = partial_index_path_map["output_index_path"]
        input_index_paths = partial_index_path_map["input_index_paths"]

        if not os.path.isfile(output_index_path):

            assert len(input_index_paths) >= 2, \
                "if singular input index, path should already exist."

            # Output index.
            output_index = faiss.read_index(empty_index_path)
            output_invlists = output_index.invlists

            # Merge input indexes.
            for input_iter, input_index_path in enumerate(input_index_paths):

                assert input_index_path is not None, "edge case."

                timer.push("read")
                input_index = faiss.read_index(input_index_path)
                input_invlists = input_index.invlists
                timer.pop()

                print_rank("ivfpq / add / merge, input %d / %d. [ +%d -> %d ]" % (
                    input_iter,
                    len(input_index_paths),
                    input_index.ntotal,
                    input_index.ntotal + output_index.ntotal,
                ))

                timer.push("add")
                for list_id in range(input_invlists.nlist):
                    output_invlists.add_entries(
                        list_id,
                        input_invlists.list_size(list_id),
                        input_invlists.get_ids(list_id),
                        input_invlists.get_codes(list_id),
                    )
                timer.pop()

                output_index.ntotal += input_index.ntotal

            timer.push("write")
            faiss.write_index(output_index, output_index_path)
            timer.pop()

        # Delete input indexes.
        # raise Exception("delete input files.")
        if len(input_index_paths) >= 2:
            timer.push("delete")
            # for path in enumerate(input_index_paths):
            for path in input_index_paths:
                # delete_this_flipping_file(input_index_path)
                os.remove(path)
            timer.pop()

    def add(self, input_data_paths, dir_path, timer):

        # Num batches & rows.
        num_batches = len(input_data_paths)
        # num_batches = 47000 # ... ~15.52 rows
        num_rows = self.get_num_rows(num_batches)

        # Missing index paths.
        missing_index_paths = self.get_missing_index_paths(
            input_data_paths,
            dir_path,
            timer,
            num_rows,
        )

        torch.distributed.barrier() # prevent race condition for missing paths

        # Iterate rows
        for row in range(num_rows):

            # timer.push("row %d of %d" % (row, num_rows))
            timer.push("row-%d" % row)

            num_cols = self.get_num_cols(num_batches, row)
            # for col in range(rank, num_batches, world_size * int(2**row)):
            for col in range(num_cols):

                # timer.push("col")

                print_rank(0, "r %d / %d, c %d / %d." % (
                    row,
                    num_rows,
                    col,
                    num_cols,
                ))

                # Input/output index paths.
                partial_index_path_map = self.get_partial_index_path_map(
                    input_data_paths,
                    dir_path,
                    row,
                    col,
                )

                # Handle edge cases.
                if partial_index_path_map is None or \
                   partial_index_path_map["output_index_path"] not in \
                   missing_index_paths:
                    continue

                # Initialize/merge partial indexes.
                if row == 0:
                    timer.push("init-partial")
                    self.init_partial(partial_index_path_map, dir_path, timer)
                    timer.pop()
                else:
                    timer.push("merge-partial")
                    self.merge_partial(partial_index_path_map, dir_path, timer)
                    timer.pop()

                # timer.pop()

            torch.distributed.barrier() # prevent inter-row race condition.

            timer.pop()

        torch.distributed.barrier() # unnecessary?

    @classmethod
    def time_merge_partials(cls, args, timer):
    
        from retro.utils import Timer
        timer = Timer()

        get_cluster_ids = lambda n : np.random.randint(
            args.ncluster,
            size = (n, 1),
            dtype = "i8",
        )

        # Num batches & rows.
        # ntrain = int(10e6)
        batch_size = int(1e6)
        num_batches = 8192 # 1024 # 10
        num_rows = cls.get_num_rows(num_batches)

        # torch.distributed.barrier() # prevent race condition for missing paths

        empty_index_path = "/mnt/fsx-outputs-chipdesign/lmcafee/retro/index/faiss-decomp-rand-100k/OPQ32_256,IVF1048576_HNSW32,PQ32__t3000000/cluster/ivfpq/empty.faissindex"
        # input_index_path = empty_index_path
        # index = faiss.read_index(index_path)

        # data = np.random.rand(batch_size, args.nfeats).astype("f4")
        data = np.random.rand(batch_size, args.ivf_dim).astype("f4")

        # Iterate rows
        for row in range(10, num_rows):

            timer.push("row-%d" % row)

            num_cols = cls.get_num_cols(num_batches, row)

            print_rank(0, "r %d / %d, c -- / %d." % (
                row,
                num_rows,
                num_cols,
            ))

            input_index_path = os.path.join(
                "/mnt/fsx-outputs-chipdesign/lmcafee/retro/index/tmp",
                "index-r%03d.faissindex" % (row - 1),
            )
            output_index_path = os.path.join(
                "/mnt/fsx-outputs-chipdesign/lmcafee/retro/index/tmp",
                "index-r%03d.faissindex" % row,
            )

            # Initialize/merge partial indexes.
            if row == 0:
                timer.push("init-partial")

                timer.push("read")
                index = faiss.read_index(empty_index_path)
                # self.c_verbose(index, True) # too much verbosity, with batch 1M
                # self.c_verbose(index.quantizer, True)
                timer.pop()

                timer.push("cluster-ids")
                cluster_ids = get_cluster_ids(len(data))
                timer.pop()

                timer.push("add-core")
                index.add_core(
                    n = len(data),
                    x = self.swig_ptr(data),
                    xids = self.swig_ptr(np.arange(len(data), dtype = "i8")),
                    precomputed_idx = self.swig_ptr(cluster_ids),
                )
                timer.pop()

                timer.pop()

            else:
                # timer.push("merge-partial")
                # self.merge_partial(partial_index_path_map, dir_path, timer)

                # Output index.
                timer.push("read-output")
                # output_index = faiss.read_index(empty_index_path)
                output_index = faiss.read_index(input_index_path)
                output_invlists = output_index.invlists
                timer.pop()

                # Merge input indexes.
                # for input_iter in range(2):
                for input_iter in range(1): # output initialized w/ first input

                    timer.push("read-input")
                    input_index = faiss.read_index(input_index_path)
                    input_invlists = input_index.invlists
                    timer.pop()

                    # # timer.push("cluster-ids")
                    # cluster_ids = get_cluster_ids(input_index.ntotal)
                    # # timer.pop()

                    print_rank("ivfpq / merge, input %d / 2. [ +%d -> %d ]"%(
                        input_iter,
                        input_index.ntotal,
                        input_index.ntotal + output_index.ntotal,
                    ))

                    timer.push("add-entry")
                    id_start = output_index.ntotal
                    for list_id in range(input_invlists.nlist):
                        input_list_size = input_invlists.list_size(list_id)
                        if input_list_size == 0:
                            continue
                        ids = self.swig_ptr(np.arange(
                            # output_index.ntotal + input_index.ntotal,
                            id_start,
                            id_start + input_list_size,
                            dtype = "i8",
                        ))
                        output_invlists.add_entries(
                            list_id,
                            input_list_size,
                            # input_invlists.get_ids(list_id),
                            ids,
                            input_invlists.get_codes(list_id),
                        )
                        id_start += input_list_size
                    timer.pop()

                    # output_index.ntotal += input_index.ntotal
                    output_index.ntotal = id_start

                index = output_index

            timer.push("write")
            faiss.write_index(index, output_index_path)
            timer.pop()

            timer.pop()

        timer.print()
        exit(0)
