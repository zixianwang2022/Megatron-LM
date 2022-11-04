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

"""Parallel version of Faiss's index.add()."""

import faiss
import numpy as np
import os
import torch

from megatron import get_args, print_rank_0
# from tools.bert_embedding import embed_text_datasets
from tools.bert_embedding import BertEmbedder
# from tools.retro.data import load_data
from tools.retro.index import Index
from tools.retro.index.indexes.faiss_base import FaissBaseIndex

# >>>
from lutil import pax, print_seq
# <<<


# def get_dataset_sample_ranges(num_samples):
def get_dataset_block_ranges(num_samples):
    args = get_args()
    block_size = args.retro_block_size
    start_idxs = list(range(0, num_samples, block_size))
    end_idxs = [min(num_samples, s + block_size) for s in start_idxs]
    ranges = list(zip(start_idxs, end_idxs))
    return ranges


def get_num_rows(num_blocks):
    return int(np.ceil(np.log(num_blocks) / np.log(2))) + 1


def get_num_cols(num_blocks, row):
    world_size = torch.distributed.get_world_size()
    return int(np.ceil(num_blocks / world_size / 2**row))


class FaissParallelAddIndex(Index):

    # def train(self, input_data_paths, dir_path, timer):
    #     # raise Exception("better to inherit from FaissBaseIndex?")
    #     index = FaissBaseIndex() # self.args)
    #     return index.train(input_data_paths, dir_path, timer)
    def train(self, *args):
        return FaissBaseIndex().train(*args)


    def get_partial_index_path_map(
            self,
            # input_data_paths,
            # num_blocks,
            dataset_sample_ranges,
            dir_path,
            row,
            col,
            rank = None,
    ):

        # pax(0, {"dir_path": dir_path})

        rank = torch.distributed.get_rank() if rank is None else rank
        world_size = torch.distributed.get_world_size()
        # num_blocks = len(input_data_paths)
        num_blocks = len(dataset_sample_ranges)

        block_str_len = int(np.ceil(np.log(num_blocks) / np.log(10))) + 1
        zf = lambda b : str(b).zfill(block_str_len)

        # First block id.
        block_id_0 = 2**row * (rank + col * world_size)

        if block_id_0 >= num_blocks:
            return None

        # Other block ids.
        if row == 0:
            block_id_1 = block_id_2 = block_id_3 = block_id_0
        else:
            block_full_range = 2**row
            block_half_range = int(block_full_range / 2)
            block_id_1 = block_id_0 + block_half_range - 1
            block_id_2 = block_id_1 + 1
            block_id_3 = block_id_2 + block_half_range - 1

        # Block ranges.
        def get_block_range(b0, b1):
            if b1 >= num_blocks:
                b1 = num_blocks - 1
            if b0 >= num_blocks:
                return None
            return b0, b1

        def block_range_to_index_path(_row, _range):
            if _range is None:
                return None
            else:
                return os.path.join(
                    dir_path,
                    "added_%s_%s-%s.faissindex" % (
                        zf(_range[-1] - _range[0] + 1),
                        *[zf(b) for b in _range],
                    ),
                )

        input_block_ranges = [
            get_block_range(block_id_0, block_id_1),
            get_block_range(block_id_2, block_id_3),
        ]
        output_block_range = get_block_range(block_id_0, block_id_3)

        # Path map.
        path_map = {
            "block_id" : block_id_0,
            "num_blocks" : num_blocks,
            "output_index_path" :
            block_range_to_index_path(row, output_block_range),
        }
        if row == 0:
            # path_map["input_data_path"] = input_data_paths[block_id_0]
            path_map["dataset_sample_range"] = dataset_sample_ranges[block_id_0]
        else:
            input_index_paths = \
                [block_range_to_index_path(row-1, r) for r in input_block_ranges]
            input_index_paths = [ p for p in input_index_paths if p is not None ]

            if not input_index_paths:
                return None

            path_map["input_index_paths"] = input_index_paths

        # Return.
        return path_map


    def get_missing_index_paths(self, dataset_sample_ranges, num_rows, dir_path):

        world_size = torch.distributed.get_world_size()
        # num_blocks, num_rows = self.get_num_blocks_and_rows(n_samples)
        num_blocks = len(dataset_sample_ranges)

        missing_index_paths = set()
        for row in range(num_rows - 1, -1, -1):

            num_cols = get_num_cols(num_blocks, row)
            for col in range(num_cols):

                for rank in range(world_size):

                    # Input/output index paths.
                    path_map = self.get_partial_index_path_map(
                        # input_data_paths,
                        # num_blocks,
                        dataset_sample_ranges,
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

        # pax(0, {"missing_index_paths": missing_index_paths})

        return missing_index_paths


    def get_added_index_path(self, dataset_sample_ranges, dir_path):
        num_rows = get_num_rows(len(dataset_sample_ranges))
        index_path_map = self.get_partial_index_path_map(
            dataset_sample_ranges,
            dir_path,
            row = num_rows - 1,
            col = 0,
            rank = 0,
        )
        return index_path_map["output_index_path"]


    # def embed_partial(self, partial_index_path_map, text_dataset, embed_dir):
    def embed_partial(self, partial_index_path_map, text_dataset, embedder):

        # args = get_args()

        sub_dataset = torch.utils.data.Subset(
            text_dataset,
            range(*partial_index_path_map["dataset_sample_range"]),
        )

        # clear_embedding_dir(EMBED_KEY)
        # embed_text_datasets(
        #     {"index": {
        #         "data" : subset_dataset,
        #         "embed_dir" : embed_dir,
        #     }},
        #     args.retro_bert_max_chunk_length,
        #     args.retro_block_size,
        #     # len(subset_dataset),
        # )
        embeddings = embedder.embed_text_dataset(sub_dataset, len(text_dataset))

        # pax(0, {
        #     "partial_index_path_map" : partial_index_path_map,
        #     "text_dataset / len" : len(text_dataset),
        #     "sub_dataset / len" : len(sub_dataset),
        #     "embedder" : embedder,
        #     "embeddings" : embeddings,
        # })

        return embeddings


    def encode_partial(self, partial_index_path_map, dir_path, timer,
                       text_dataset, embedder):
        """Encode partial indexes (embarrassingly parallel).

        Encode the partial indexes, generally in blocks of 1M vectors each.
        For each block, the empty/trained index is loaded, and index.add() is
        called on each block of data.
        """

        # Index & data paths.
        empty_index_path = self.get_empty_index_path(dir_path)
        partial_index_path = partial_index_path_map["output_index_path"]
        # input_data_path = partial_index_path_map["input_data_path"]

        # If partial index exists, return.
        if os.path.isfile(partial_index_path):
            return

        # >>>
        # pax(0, {
        #     "partial_index_path_map" : partial_index_path_map,
        #     "text_dataset" : text_dataset,
        #     "empty_index_path" : empty_index_path,
        #     "partial_index_path" : partial_index_path,
        # })
        # print_seq("partial_index_path = %s." % partial_index_path)
        # <<<

        # Embed data block.
        # embed_dir = os.path.join(
        #     dir_path,
        #     "embed_%d" % partial_index_path_map["block_id"],
        # )
        # self.embed_partial(partial_index_path_map, text_dataset, embed_dir)
        timer.push("embed-data")
        input_data = self.embed_partial(partial_index_path_map,
                                        text_dataset, embedder)
        timer.pop()

        # pax(0, {"input_data": input_data})

        # # Load data block.
        # timer.push("load-data")
        # input_data = load_data([input_data_path], timer)["data"].astype("f4")
        # timer.pop()

        nvecs = len(input_data)
        print_rank_0("ivfpq / add / partial,  block %d / %d. [ %d vecs ]" % (
            partial_index_path_map["block_id"],
            partial_index_path_map["num_blocks"],
            nvecs,
        ))

        timer.push("read")
        index = faiss.read_index(empty_index_path)
        # self.c_verbose(index, True) # with block_size 1M ... too fast/verbose
        # self.c_verbose(index.quantizer, True)
        timer.pop()

        timer.push("add")
        index.add(input_data)
        timer.pop()

        timer.push("write")
        faiss.write_index(index, partial_index_path)
        timer.pop()


    def merge_partial(self, partial_index_path_map, dir_path, timer):
        '''Merge partial indexes.

        Pairwise merging
        '''

        # Extract inverted lists from full index.
        def get_invlists(index):
            return faiss.extract_index_ivf(index).invlists

        # Index paths.
        output_index_path = partial_index_path_map["output_index_path"]
        input_index_paths = partial_index_path_map["input_index_paths"]

        if not os.path.isfile(output_index_path):

            assert len(input_index_paths) >= 2, \
                "if singular input index, path should already exist."

            # Init output index.
            timer.push("read/init-output")
            output_index = faiss.read_index(input_index_paths[0])
            output_invlists = get_invlists(output_index)
            timer.pop()

            # Merge input indexes.
            for input_iter in range(1, len(input_index_paths)):

                input_index_path = input_index_paths[input_iter]
                assert input_index_path is not None, "missing input index."

                timer.push("read-input")
                input_index = faiss.read_index(input_index_path)
                input_invlists = get_invlists(input_index)
                timer.pop()

                print_rank_0("ivfpq / add / merge, input %d / %d. [ +%d -> %d ]" % (
                    input_iter,
                    len(input_index_paths),
                    input_index.ntotal,
                    input_index.ntotal + output_index.ntotal,
                ))

                timer.push("add")

                # Old way.
                # for list_id in range(input_invlists.nlist):
                #     output_invlists.add_entries(
                #         list_id,
                #         input_invlists.list_size(list_id),
                #         input_invlists.get_ids(list_id),
                #         input_invlists.get_codes(list_id),
                #     )

                # New way.
                output_invlists.merge_from(input_invlists, output_index.ntotal)

                timer.pop()

                output_index.ntotal += input_index.ntotal

            timer.push("write")
            faiss.write_index(output_index, output_index_path)
            timer.pop()

        # Delete input indexes.
        if len(input_index_paths) >= 2:
            timer.push("delete")
            for path in input_index_paths:
                os.remove(path)
            timer.pop()


    # def add(self, input_data_paths, dir_path, timer):
    def add(self, text_dataset, dir_path, timer):
        """Add vectors to index, in parallel.

        Two stage process:
        1. Encode partial indexes (i.e., starting with empty index, encode
           blocks of 1M samples per partial index).
        2. Merge partial indexes.
           - This is a pairwise hierarchical merge.
           - We iterate log2(num_blocks) 'rows' of merge.
           - Ranks move in lock-step across each row (i.e., 'cols')
        """

        args = get_args()

        # Set OMP threads (torch defaults to n_threads = 1).
        faiss.omp_set_num_threads(4)
        # pax(0, {"nthreads": faiss.omp_get_max_threads()})

        # Num blocks & rows.
        dataset_sample_ranges = get_dataset_block_ranges(len(text_dataset))
        num_blocks = len(dataset_sample_ranges)
        num_rows = get_num_rows(num_blocks)

        # Missing index paths.
        # missing_blocks = self.get_missing_blocks(
        missing_index_paths = self.get_missing_index_paths(
            dataset_sample_ranges,
            num_rows,
            dir_path,
        )

        # pax(0, {"missing_blocks": missing_blocks})

        # Prevent race condition for missing paths. [ necessary? ]
        torch.distributed.barrier()

        # Bert embedder.
        embedder = BertEmbedder(args.retro_bert_max_chunk_length)
        # pax(0, {"embedder": embedder})

        # Iterate merge rows.
        for row in range(num_rows):

            timer.push("row-%d" % row)

            # Get number of columns for this merge row.
            num_cols = get_num_cols(num_blocks, row)

            # Iterate merge columns.
            for col in range(num_cols):

                print_rank_0("r %d / %d, c %d / %d." % (
                    row,
                    num_rows,
                    col,
                    num_cols,
                ))

                # Input/output index paths.
                partial_index_path_map = self.get_partial_index_path_map(
                    # input_data_paths,
                    # text_dataset,
                    dataset_sample_ranges,
                    dir_path,
                    row,
                    col,
                )

                # >>>
                # pax(0, {"partial_index_path_map": partial_index_path_map})
                # print_seq("block %d, range %s." % (
                #     partial_index_path_map["block_id"],
                #     partial_index_path_map["dataset_sample_range"],
                # ))
                # <<<

                # Handle edge cases.
                if partial_index_path_map is None or \
                   partial_index_path_map["output_index_path"] not in \
                   missing_index_paths:
                    continue

                # Initialize/merge partial indexes.
                if row == 0:
                    timer.push("init-partial")
                    self.encode_partial(partial_index_path_map, dir_path, timer,
                                        text_dataset, embedder)
                    timer.pop()
                else:
                    timer.push("merge-partial")
                    self.merge_partial(partial_index_path_map, dir_path, timer)
                    timer.pop()

            # Prevent inter-row race condition.
            torch.distributed.barrier()

            timer.pop()

        # Final barrier. [ necessary? ]
        torch.distributed.barrier()

        # Get output index path, for return.
        output_index_path = self.get_added_index_path(dataset_sample_ranges,
                                                      dir_path)
        # pax(0, {"output_index_path": output_index_path})

        return output_index_path


    # @classmethod
    # def time_hnsw(cls, args, timer):
    #     """Timing model for HNSW cluster assignment."""

    #     if torch.distributed.get_rank() != 0:
    #         return

    #     from lutil import pax
    #     from tools.retro.utils import Timer

    #     # pax({"max threads": faiss.omp_get_max_threads()})

    #     timer = Timer()

    #     timer.push("read-index")
    #     empty_index_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/index/faiss-base-corpus-clean/OPQ32_256,IVF4194304_HNSW32,PQ32__t100000000__trained.faissindex"
    #     index = faiss.read_index(empty_index_path)
    #     index_ivf = faiss.extract_index_ivf(index)
    #     quantizer = index_ivf.quantizer
    #     timer.pop()

    #     block_sizes = [ int(a) for a in [ 1e3, 1e6 ] ]
    #     nprobes = 1, 128, 1024, 4096 # 66000

    #     # >>>
    #     # if 1:
    #     #     data = np.random.rand(10, args.ivf_dim).astype("f4")
    #     #     D1, I1 = quantizer.search(data, 1)
    #     #     D2, I2 = quantizer.search(data, 2)
    #     #     D128, I128 = quantizer.search(data, 128)
    #     #     # print(np.vstack([ I1[:,0], D1[:,0] ]).T)
    #     #     # print(np.vstack([ I2[:,0], D2[:,0] ]).T)
    #     #     # print(np.vstack([ I128[:,0], D128[:,0] ]).T)
    #     #     print(np.vstack([ I1[:,0], I2[:,0], I128[:,0] ]).T)
    #     #     print(np.vstack([ D1[:,0], D2[:,0], D128[:,0] ]).T)
    #     #     # print(I1[:,0])
    #     #     # print(I2)
    #     #     # print(I128)
    #     #     # print(D1)
    #     #     # print(D2)
    #     #     # print(D128)
    #     #     exit(0)
    #     # <<<

    #     for block_size_index, block_size in enumerate(block_sizes):

    #         timer.push("data-%d" % block_size)
    #         data = np.random.rand(block_size, args.ivf_dim).astype("f4")
    #         timer.pop()

    #         for nprobe_index, nprobe in enumerate(nprobes):

    #             timer.push("search-%d-%d" % (block_size, nprobe))
    #             D, I = quantizer.search(data, nprobe)
    #             timer.pop()

    #             # if nprobe > 1:
    #             #     D1, I1 = quantizer.search(data, 1)
    #             #     pax({
    #             #         "I1" : I1,
    #             #         "I" : I,
    #             #     })

    #             print("time hnsw ... bs %d [ %d/%d ]; nprobe %d [ %d/%d ]." % (
    #                 block_size,
    #                 block_size_index,
    #                 len(block_sizes),
    #                 nprobe,
    #                 nprobe_index,
    #                 len(nprobes),
    #             ))

    #     timer.print()
    #     exit(0)

    #     pax(0, {
    #         "index" : index,
    #         "index_ivf" : index_ivf,
    #         "quantizer" : quantizer,
    #         "result" : result,
    #     })


    # @classmethod
    # def time_query(cls, args, timer):
    #     """Timing model for querying."""

    #     if torch.distributed.get_rank() != 0:
    #         return

    #     from lutil import pax
    #     from tools.retro.utils import Timer

    #     # >>>
    #     faiss.omp_set_num_threads(1) # 128)
    #     # pax({"max threads": faiss.omp_get_max_threads()})
    #     # <<<

    #     timer = Timer()

    #     timer.push("read-index")
    #     added_index_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/index/faiss-par-add-corpus-clean/OPQ32_256,IVF4194304_HNSW32,PQ32__t100000000/added_064_000-063.faissindex"
    #     index = faiss.read_index(added_index_path)
    #     index_ivf = faiss.extract_index_ivf(index)
    #     timer.pop()

    #     block_sizes = [ int(a) for a in [ 1e2, 1e4 ] ]
    #     # nprobes = 1, 16, 128, 1024, 4096 # 66000
    #     # nprobes = 2, 4
    #     nprobes = 4096, # 16, 128
        
    #     # pax({"index": index})

    #     for block_size_index, block_size in enumerate(block_sizes):

    #         timer.push("data-%d" % block_size)
    #         opq_data = np.random.rand(block_size, args.nfeats).astype("f4")
    #         timer.pop()

    #         for nprobe_index, nprobe in enumerate(nprobes):

    #             nnbr = 100

    #             timer.push("search-%d-%d" % (block_size, nprobe))

    #             # >>>
    #             index_ivf.nprobe = nprobe
    #             # pax({
    #             #     "index_ivf" : index_ivf,
    #             #     "quantizer" : index_ivf.quantizer,
    #             # })
    #             # <<<

    #             timer.push("full")
    #             index.search(opq_data, nnbr)
    #             timer.pop()

    #             timer.push("split")

    #             timer.push("preproc")
    #             ivf_data = index.chain.at(0).apply(opq_data)
    #             timer.pop()

    #             timer.push("assign")
    #             D_hnsw, I_hnsw = index_ivf.quantizer.search(ivf_data, nprobe)
    #             timer.pop()

    #             # timer.push("pq")
    #             # I = np.empty((block_size, nnbr), dtype = "i8")
    #             # D = np.empty((block_size, nnbr), dtype = "f4")
    #             # index_ivf.search_preassigned(
    #             #     block_size,
    #             #     # cls.swig_ptr(I[:,0]),
    #             #     cls.swig_ptr(ivf_data),
    #             #     nnbr,
    #             #     cls.swig_ptr(I_hnsw), # [:,0]),
    #             #     cls.swig_ptr(D_hnsw), # [:,0]),
    #             #     cls.swig_ptr(D),
    #             #     cls.swig_ptr(I),
    #             #     False,
    #             # )
    #             # timer.pop()

    #             timer.push("swig")
    #             I = np.empty((block_size, nnbr), dtype = "i8")
    #             D = np.empty((block_size, nnbr), dtype = "f4")
    #             ivf_data_ptr = cls.swig_ptr(ivf_data)
    #             I_hnsw_ptr = cls.swig_ptr(I_hnsw)
    #             D_hnsw_ptr = cls.swig_ptr(D_hnsw)
    #             D_ptr = cls.swig_ptr(D)
    #             I_ptr = cls.swig_ptr(I)
    #             timer.pop()

    #             timer.push("prefetch")
    #             index_ivf.invlists.prefetch_lists(I_hnsw_ptr, block_size * nprobe)
    #             timer.pop()

    #             timer.push("search-preassign")
    #             index_ivf.search_preassigned(
    #                 block_size,
    #                 ivf_data_ptr,
    #                 nnbr,
    #                 I_hnsw_ptr,
    #                 D_hnsw_ptr,
    #                 D_ptr,
    #                 I_ptr,
    #                 True, # False,
    #             )
    #             timer.pop()

    #             timer.pop()

    #             # pax({"I": I, "D": D})

    #             # print("time query ... bs %d [ %d/%d ]; nprobe %d [ %d/%d ]." % (
    #             #     block_size,
    #             #     block_size_index,
    #             #     len(block_sizes),
    #             #     nprobe,
    #             #     nprobe_index,
    #             #     len(nprobes),
    #             # ))

    #             timer.pop()

    #     timer.print()
    #     exit(0)

    #     pax(0, {
    #         "index" : index,
    #         "index_ivf" : index_ivf,
    #         "quantizer" : quantizer,
    #         "result" : result,
    #     })


    # @classmethod
    # def time_merge_partials(cls, args, timer):
    #     """Timing model for merging partial indexes."""
    
    #     if torch.distributed.get_rank() != 0:
    #         return

    #     from retro.utils import Timer
    #     timer = Timer()

    #     get_cluster_ids = lambda n : np.random.randint(
    #         args.ncluster,
    #         size = (n, 1),
    #         dtype = "i8",
    #     )

    #     # Num blocks & rows.
    #     block_size = int(1e6)
    #     num_blocks = 8192 # 1024 # 10
    #     num_rows = get_num_rows(num_blocks)

    #     raise Exception("switch to IVF4194304.")
    #     empty_index_path = "/mnt/fsx-outputs-chipdesign/lmcafee/retro/index/faiss-decomp-rand-100k/OPQ32_256,IVF1048576_HNSW32,PQ32__t3000000/cluster/ivfpq/empty.faissindex"

    #     data = np.random.rand(block_size, args.ivf_dim).astype("f4")

    #     # Iterate rows
    #     for row in range(10, num_rows):

    #         timer.push("row-%d" % row)

    #         num_cols = get_num_cols(num_blocks, row)

    #         print_rank(0, "r %d / %d, c -- / %d." % (
    #             row,
    #             num_rows,
    #             num_cols,
    #         ))

    #         input_index_path = os.path.join(
    #             "/mnt/fsx-outputs-chipdesign/lmcafee/retro/index/tmp",
    #             "index-r%03d.faissindex" % (row - 1),
    #         )
    #         output_index_path = os.path.join(
    #             "/mnt/fsx-outputs-chipdesign/lmcafee/retro/index/tmp",
    #             "index-r%03d.faissindex" % row,
    #         )

    #         # Initialize/merge partial indexes.
    #         if row == 0:
    #             timer.push("init-partial")

    #             timer.push("read")
    #             index = faiss.read_index(empty_index_path)
    #             # self.c_verbose(index, True) # too much verbosity, with block 1M
    #             # self.c_verbose(index.quantizer, True)
    #             timer.pop()

    #             timer.push("cluster-ids")
    #             cluster_ids = get_cluster_ids(len(data))
    #             timer.pop()

    #             timer.push("add-core")
    #             index.add_core(
    #                 n = len(data),
    #                 x = self.swig_ptr(data),
    #                 xids = self.swig_ptr(np.arange(len(data), dtype = "i8")),
    #                 precomputed_idx = self.swig_ptr(cluster_ids),
    #             )
    #             timer.pop()

    #             timer.pop()

    #         else:

    #             # Output index.
    #             timer.push("read-output")
    #             output_index = faiss.read_index(input_index_path)
    #             output_invlists = output_index.invlists
    #             timer.pop()

    #             # Merge input indexes.
    #             for input_iter in range(1): # output initialized w/ first input

    #                 timer.push("read-input")
    #                 input_index = faiss.read_index(input_index_path)
    #                 input_invlists = input_index.invlists
    #                 timer.pop()

    #                 # # timer.push("cluster-ids")
    #                 # cluster_ids = get_cluster_ids(input_index.ntotal)
    #                 # # timer.pop()

    #                 print_rank("ivfpq / merge, input %d / 2. [ +%d -> %d ]"%(
    #                     input_iter,
    #                     input_index.ntotal,
    #                     input_index.ntotal + output_index.ntotal,
    #                 ))

    #                 timer.push("add-entry")
    #                 id_start = output_index.ntotal
    #                 for list_id in range(input_invlists.nlist):
    #                     input_list_size = input_invlists.list_size(list_id)
    #                     if input_list_size == 0:
    #                         continue
    #                     ids = self.swig_ptr(np.arange(
    #                         # output_index.ntotal + input_index.ntotal,
    #                         id_start,
    #                         id_start + input_list_size,
    #                         dtype = "i8",
    #                     ))
    #                     # output_invlists.add_entries(
    #                     #     list_id,
    #                     #     input_list_size,
    #                     #     # input_invlists.get_ids(list_id),
    #                     #     ids,
    #                     #     input_invlists.get_codes(list_id),
    #                     # )
    #                     output_invlists.merge_from(
    #                         input_invlists,
    #                         output_index.ntotal,
    #                     )
    #                     id_start += input_list_size
    #                 timer.pop()

    #                 # output_index.ntotal += input_index.ntotal
    #                 output_index.ntotal = id_start

    #             index = output_index

    #         timer.push("write")
    #         faiss.write_index(index, output_index_path)
    #         timer.pop()

    #         timer.pop()

    #     timer.print()
    #     exit(0)
