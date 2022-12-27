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
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.retro.index import Index
from tools.retro.index.utils import num_samples_to_block_ranges


class FaissBaseIndex(Index):

    def _train(self, input_data_loader):
        '''Train index (rank 0's method).'''

        args = get_retro_args()

        assert torch.distributed.get_rank() == 0

        # Set num threads (torch.distributed reset it to 1).
        # >>>
        faiss.omp_set_num_threads(64)
        # faiss.omp_set_num_threads(32)
        # faiss.omp_set_num_threads(128)
        # <<<

        empty_index_path = self.get_empty_index_path()

        # Index already exists? -> return.
        if os.path.isfile(empty_index_path):
            return

        # Load data.
        inp = input_data_loader()

        # >>>
        # faiss.omp_set_num_threads(32)
        # faiss.omp_set_num_threads(8)
        # <<<

        # Init index.
        index = faiss.index_factory(args.retro_nfeats, args.retro_index_str)

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
        index.train(inp)

        # Save index.
        faiss.write_index(index, empty_index_path)


    def train(self, input_data_loader):
        '''Train index.'''

        # Single process only.
        if torch.distributed.get_rank() == 0:
            self._train(input_data_loader)

        torch.distributed.barrier()


    def _add(self, text_dataset, dataset_sample_ranges):
        '''Add to index (rank 0's method).'''

        assert torch.distributed.get_rank() == 0

        args = get_retro_args()

        # Set num threads (torch.distributed reset it to 1).
        faiss.omp_set_num_threads(64)

        # Bert embedder.
        embedder = BertEmbedder(args.retro_bert_batch_size,
                                args.retro_bert_max_chunk_length)

        # Empty/added index paths.
        empty_index_path = self.get_empty_index_path()
        added_index_path = self.get_added_index_path(dataset_sample_ranges)

        # Skip adding, if index exists.
        if os.path.isfile(added_index_path):
            return

        # Read trained index.
        index = faiss.read_index(empty_index_path)

        # Iterate data blocks & add.
        for sample_range in tqdm(dataset_sample_ranges, "faiss_base.add"):

            # Embed text.
            embeds = self.embed_text_dataset_block(
                embedder, text_dataset, sample_range)

            # Add to index.
            index.add(embeds)

        # Write index.
        faiss.write_index(index, added_index_path)


    def add(self, text_dataset):
        '''Add to index.'''

        dataset_sample_ranges = num_samples_to_block_ranges(len(text_dataset))

        # Single process only.
        if torch.distributed.get_rank() == 0:
            self._add(text_dataset, dataset_sample_ranges)

        # Wait for rank 0.
        torch.distributed.barrier()

        # Get output index path, for return.
        return self.get_added_index_path(dataset_sample_ranges)
