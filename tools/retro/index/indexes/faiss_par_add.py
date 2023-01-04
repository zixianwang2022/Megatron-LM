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
import h5py
import numpy as np
import os
import shutil
import torch
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding import BertEmbedder
from tools.bert_embedding.utils import get_missing_blocks_by_rank
from tools.retro.index import Index
from tools.retro.index.indexes.faiss_base import FaissBaseIndex
from tools.retro.index.utils import get_added_codes_dir, get_added_code_paths

# >>>
from lutil import pax, print_seq
# <<<


class FaissParallelAddIndex(Index):

    def train(self, *args):
        '''Use Faiss-base for training.'''
        return FaissBaseIndex().train(*args)


    def encode_block(self, index, embedder, text_dataset, block):
        '''Encode sub-dataset block, to be later added to index.

        Encode the data subset, generally in blocks of 1M vectors each. For
        each block, the empty/trained index is loaded, codes are computed
        via index.sa_encode(), and the resulting codes are saved to disk.
        '''

        args = get_retro_args()

        # Embed block.
        embeddings = self.embed_text_dataset_block(
            embedder,
            text_dataset,
            block["range"],
        )

        # Encode block.
        print_rank_0("encode.")
        codes = index.sa_encode(embeddings)

        # Save neighbors.
        print_rank_0("save codes.")
        os.makedirs(os.path.dirname(block["path"]), exist_ok = True)
        with h5py.File(block["path"], "w") as f:
            f.create_dataset("data", data = codes)


    def encode(self, text_dataset):
        '''Encode text dataset, to be later added to index.'''

        args = get_retro_args()
        codes_dir = get_added_codes_dir()

        # Index.
        index = self.get_empty_index()

        # Bert embedder.
        embedder = BertEmbedder(args.retro_bert_batch_size,
                                args.retro_bert_max_chunk_length,
                                args.bert_embedder_type)

        # Missing code blocks.
        def validate(f):
            assert len(f["data"].shape) == 2
        n_missing_blocks, missing_code_blocks = get_missing_blocks_by_rank(
            codes_dir,
            len(text_dataset),
            args.retro_block_size,
            validate = validate,
        )

        # Encode each block.
        for block_index, block in enumerate(missing_code_blocks):

            if block is not None:

                # Progress.
                print_rank_0("encode block %d / %d ... %s." % (
                    block_index,
                    len(missing_code_blocks),
                    block["path"],
                ))

                # Query block neighbors.
                self.encode_block(index, embedder, text_dataset, block)

            # Synchronize progress across all ranks. (for easier observation)
            print_rank_0(" > waiting for other ranks to finish block.")
            torch.distributed.barrier()


    # def add_codes(self, text_dataset):
    def add_codes(self):

        assert torch.distributed.get_rank() == 0

        added_index_path = self.get_added_index_path()
        if os.path.exists(added_index_path):
            return

        # Index.
        print_rank_0("read empty index.")
        index = self.get_empty_index()
        index_ivf = faiss.extract_index_ivf(index)

        # Add codes.
        print_rank_0("add codes.")
        code_paths = get_added_code_paths()
        # >>>
        pax(0, {"code_paths": code_paths})
        # <<<
        for code_path in tqdm(code_paths, "add codes"):
            with h5py.File(code_path) as f:
                codes = np.copy(f["data"])
                index_ivf.add_sa_codes(codes)

        # Write index.
        print_rank_0("write added index.")
        faiss.write_index(index, added_index_path)


    def remove_codes(self):
        '''Remove added codes after adding to index.'''
        assert torch.distributed.get_rank() == 0
        added_index_path = self.get_added_index_path()
        assert os.path.isfile(added_index_path)
        shutil.rmtree(get_added_codes_dir())


    def add(self, text_dataset):
        self.encode(text_dataset)
        if torch.distributed.get_rank() == 0:
            self.add_codes()
            # >>>
            # self.remove_codes() # debugging ... manually remove, for now
            # <<<
        torch.distributed.barrier()
