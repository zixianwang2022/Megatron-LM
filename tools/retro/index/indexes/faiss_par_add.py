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

"""Multi-process (& multi-node) version of Faiss's index.add()."""

import faiss
import numpy as np
import os
import torch

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.retro.index import Index
from tools.retro.index.indexes.faiss_base import FaissBaseIndex
from tools.retro.index.utils import num_samples_to_block_ranges


def get_num_rows(num_blocks):
    '''Number of synchronous, sequential steps for 2-way merging of blocks.'''
    return int(np.ceil(np.log(num_blocks) / np.log(2))) + 1


def get_num_cols(num_blocks, row):
    '''
    Number of asynchronous, parallel steps within a synchronous row
    of merges.
    '''
    world_size = torch.distributed.get_world_size()
    return int(np.ceil(num_blocks / world_size / 2**row))


class FaissParallelAddIndex(Index):

    def train(self, *args):
        '''Use Faiss-base for training.'''
        return FaissBaseIndex().train(*args)


    def get_partial_index_path_map(
            self,
            dataset_sample_ranges,
            dir_path,
            row,
            col,
            rank = None,
    ):
        '''Block & filename info for each partial index.

        Given row & col within merging grid, get the unique block/filename
        for this partial index. Also, get the names of this block's two
        input indexes required for merging.
        '''

        rank = torch.distributed.get_rank() if rank is None else rank
        world_size = torch.distributed.get_world_size()
        num_blocks = len(dataset_sample_ranges)

        # Z-fill output block filenames, for sorting.
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

        # Input index ranges for merging.
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
        '''Find missing partial indexes.'''

        world_size = torch.distributed.get_world_size()
        num_blocks = len(dataset_sample_ranges)

        # Find missing partial indexes by iterating backwards through merge grid.
        missing_index_paths = set()
        for row in range(num_rows - 1, -1, -1):

            num_cols = get_num_cols(num_blocks, row)
            for col in range(num_cols):

                # Each rank aware of blocks processed by other ranks.
                for rank in range(world_size):

                    # Input/output index paths.
                    path_map = self.get_partial_index_path_map(
                        dataset_sample_ranges,
                        dir_path,
                        row,
                        col,
                        rank,
                    )

                    # Handle edge cases.
                    if path_map is None:
                        continue

                    # Add to missing paths.
                    output_path = path_map["output_index_path"]
                    input_paths = path_map.get("input_index_paths", [])

                    if row == num_rows - 1 and not os.path.isfile(output_path):
                        missing_index_paths.add(output_path)

                    if output_path in missing_index_paths:
                        missing_input_paths = \
                            [ p for p in input_paths if not os.path.isfile(p) ]
                        missing_index_paths.update(missing_input_paths)

        return missing_index_paths


    def get_added_index_path(self, dataset_sample_ranges, dir_path):
        '''Path of final, fully constructed index.'''
        num_rows = get_num_rows(len(dataset_sample_ranges))
        index_path_map = self.get_partial_index_path_map(
            dataset_sample_ranges,
            dir_path,
            row = num_rows - 1,
            col = 0,
            rank = 0,
        )
        return index_path_map["output_index_path"]


    def encode_partial(self, partial_index_path_map, dir_path,
                       text_dataset, embedder):
        """Encode partial indexes (embarrassingly parallel).

        Encode the partial indexes, generally in blocks of 1M vectors each.
        For each block, the empty/trained index is loaded, and index.add() is
        called on each block of data.
        """

        # Index & data paths.
        empty_index_path = self.get_empty_index_path(dir_path)
        partial_index_path = partial_index_path_map["output_index_path"]

        # If partial index exists, return.
        if os.path.isfile(partial_index_path):
            return

        # Embed data block.
        input_data = self.embed_text_dataset_block(
            embedder,
            text_dataset,
            partial_index_path_map["dataset_sample_range"],
        )

        # Print progress.
        nvecs = len(input_data)
        print_rank_0("ivfpq / add / partial,  block %d / %d. [ %d vecs ]" % (
            partial_index_path_map["block_id"],
            partial_index_path_map["num_blocks"],
            nvecs,
        ))

        # Read index.
        index = faiss.read_index(empty_index_path)
        # self.c_verbose(index, True) # with block_size <1M, too verbose
        # self.c_verbose(index.quantizer, True)

        # Add to index.
        index.add(input_data)

        # Write index.
        faiss.write_index(index, partial_index_path)


    def merge_partial(self, partial_index_path_map, dir_path):
        '''Merge partial indexes.

        Pairwise merging of partial indexes. For each row in the merge grid,
        we merge two indexes from the previous row. Remove input indexes
        afterward.
        '''

        # Extract inverted lists from full index.
        def get_invlists(index):
            return faiss.extract_index_ivf(index).invlists

        # Index paths.
        output_index_path = partial_index_path_map["output_index_path"]
        input_index_paths = partial_index_path_map["input_index_paths"]

        # Merge, if not yet merged (i.e., output file doesn't exist).
        if not os.path.isfile(output_index_path):

            assert len(input_index_paths) >= 2, \
                "if singular input index, path should already exist."

            # Init output index.
            output_index = faiss.read_index(input_index_paths[0])
            output_invlists = get_invlists(output_index)

            # Merge input indexes.
            for input_iter in range(1, len(input_index_paths)):

                # Get index path.
                input_index_path = input_index_paths[input_iter]
                assert input_index_path is not None, "missing input index."

                # Read index, invlists.
                input_index = faiss.read_index(input_index_path)
                input_invlists = get_invlists(input_index)

                # Print progress.
                print_rank_0("ivfpq / add / merge, input %d / %d. [ +%d -> %d ]"%(
                    input_iter,
                    len(input_index_paths),
                    input_index.ntotal,
                    input_index.ntotal + output_index.ntotal,
                ))

                # Merge inverted lists.
                output_invlists.merge_from(input_invlists, output_index.ntotal)

                output_index.ntotal += input_index.ntotal

            # Write index.
            faiss.write_index(output_index, output_index_path)

        # Delete input indexes.
        if len(input_index_paths) >= 2:
            for path in input_index_paths:
                os.remove(path)


    def add(self, text_dataset, dir_path):
        """Add vectors to index, in parallel.

        Two stage process:
        1. Encode partial indexes (i.e., starting with empty index, encode
           blocks of 1M samples per partial index).
        2. Merge partial indexes.
           - This is a pairwise hierarchical merge.
           - We iterate log2(num_blocks) 'rows' of merge.
           - Ranks move in lock-step across each row (i.e., 'cols')
        """

        args = get_retro_args()

        # Set OMP threads (torch defaults to n_threads = 1).
        faiss.omp_set_num_threads(4) # generally, 4 threads X 8-16 processes.

        # Num blocks & rows.
        dataset_sample_ranges = num_samples_to_block_ranges(len(text_dataset))
        num_blocks = len(dataset_sample_ranges)
        num_rows = get_num_rows(num_blocks)

        # Missing index paths.
        missing_index_paths = self.get_missing_index_paths(
            dataset_sample_ranges,
            num_rows,
            dir_path,
        )

        # Prevent race condition for missing paths. [ necessary? ]
        torch.distributed.barrier()

        # Bert embedder.
        embedder = BertEmbedder(args.retro_bert_batch_size,
                                args.retro_bert_max_chunk_length)

        # Iterate merge rows.
        for row in range(num_rows):

            # Get number of columns for this merge row.
            num_cols = get_num_cols(num_blocks, row)

            # Iterate merge columns.
            for col in range(num_cols):

                # Print progress.
                print_rank_0("r %d / %d, c %d / %d." % (
                    row,
                    num_rows,
                    col,
                    num_cols,
                ))

                # Input/output index paths.
                partial_index_path_map = self.get_partial_index_path_map(
                    dataset_sample_ranges,
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
                    self.encode_partial(partial_index_path_map, dir_path,
                                        text_dataset, embedder)
                else:
                    self.merge_partial(partial_index_path_map, dir_path)

            # Prevent inter-row race condition.
            torch.distributed.barrier()

        # Final barrier. [ necessary? ]
        torch.distributed.barrier()

        # Get output index path, for return.
        return self.get_added_index_path(dataset_sample_ranges, dir_path)
