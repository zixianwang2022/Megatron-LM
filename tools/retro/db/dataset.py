# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import json
import numpy as np
import torch

from megatron import get_args, print_rank_0
from tools.retro.external_libs import h5py
from tools.retro.utils import get_gpt_tokenizer

# >>>
from collections import defaultdict
from tqdm import tqdm
from lutil import pax, np as _np
# <<<


class DBDataset(torch.utils.data.Dataset):
    '''Dataset for iterating chunks.

    Requires:
    - List of indexed datasets
    - Chunk index array, with format:
        [dataset_idx, doc_id, start_idx, end_idx, bert_length])
    '''

    def __init__(self, db_path, indexed_datasets, chunks, max_chunk_length):

        assert chunks.shape[1] == 5, "expected 5 columns (dataset_idx, " \
        "doc_idx, token_start_idx, token_end_idx, bert_chunk_length); " \
        "found %d columns." % chunks.shape[1]

        self.db_path = db_path
        self.indexed_datasets = indexed_datasets
        self.chunks = chunks
        self.doc_chunk_map = None

        self.max_chunk_length = max_chunk_length
        self.eod_token_id = get_gpt_tokenizer().eod

    def __len__(self):
        return self.chunks.shape[0]

    def __getitem__(self, chunk_id):

        # Chunk start/end indexes.
        indexed_dataset_id, doc_id, token_start_idx, token_end_idx, _ = \
            [ value.item() for value in self.chunks[chunk_id] ]
        chunk_length = token_end_idx - token_start_idx
        indexed_dataset = self.indexed_datasets[indexed_dataset_id]

        # Chunk token ids.
        token_ids = indexed_dataset.get(doc_id,
                                        offset=token_start_idx,
                                        length=chunk_length)

        # Extend chunks to max_chunk_length by padding with EOD tokens.
        if chunk_length != self.max_chunk_length:
            assert chunk_length < self.max_chunk_length, "invalid chunk len."
            token_ids = token_ids.tolist()
            token_ids += [self.eod_token_id] * \
                (self.max_chunk_length - chunk_length)

        return {
            "doc_id" : doc_id,
            "text" : np.array(token_ids, dtype=np.int64),
        }

    # >>>
    # def load_doc_offsets(self):
    # def load_doc_chunk_map(self):
    #     print("load doc offsets.")
    #     with h5py.File(self.db_path) as f:
    #         self.doc_chunk_map = defaultdict(dict)
    #         start_chunk_id = 0
    #         for dataset_id, doc_id, end_chunk_id in tqdm(f["doc_offsets"]):
    #             self.doc_chunk_map[dataset_id.item()][doc_id.item()] = \
    #                 (start_chunk_id, end_chunk_id.item())
    #             start_chunk_id = end_chunk_id.item()
    # def load_doc_chunk_map(self):
    #     print("load doc offsets.")
    #     with h5py.File(self.db_path) as f:
    #         self.doc_chunk_map = defaultdict(dict)
    #         n_docs = f["doc_offsets"].shape[0]
    #         block_size = int(1e6)
    #         start_chunk_id = 0
    #         for start_doc_id in tqdm(range(0, n_docs, block_size)):
    #             end_doc_id = min(n_docs, start_doc_id + block_size)
    #             block_doc_offsets = \
    #                 np.copy(f["doc_offsets"][start_doc_id:end_doc_id])
    #             for dataset_id, doc_id, end_chunk_id in block_doc_offsets:
    #                 self.doc_chunk_map[dataset_id.item()][doc_id.item()] = \
    #                     (start_chunk_id, end_chunk_id.item())
    #                 start_chunk_id = end_chunk_id.item()
    #         pax({"doc_chunk_map": len(self.doc_chunk_map)})

    # def get_doc_chunk_range(self, dataset_id, doc_id):
    #     assert self.doc_chunk_map, "call 'load_doc_chunk_map()' first."
    #     return self.doc_chunk_map[dataset_id][doc_id]
    # <<<

    # >>>
    # def load_doc_chunk_map(self):
    def load_doc_tuples(self):
        self.doc_tuples = np.zeros(shape=(len(self), 2), dtype="uint32")
        # self.doc_tuples = []
        # block_size = int(1e8)
        block_size = int(1e6)
        for start_idx in tqdm(range(0, len(self), block_size)):
            end_idx = min(len(self), start_idx + block_size)
            # >>>
            self.doc_tuples[start_idx:end_idx]=self.chunks[start_idx:end_idx,:2]
            # +++
            # # block = np.copy(self.chunks[start_idx:end_idx, :2])
            # block = self.chunks[start_idx:end_idx, :2]
            # [ self.doc_tuples.append(tuple(entry.tolist())) for entry in block ]
            # <<<

        # >>>
        # print("~~~")
        # # print(self.doc_tuples)
        # pax({
        #     # "db" : self,
        #     "db / len" : len(self),
        #     **{f"doc_tuples / {i}" : str(self.doc_tuples[i])
        #        for i in range(0,len(self.doc_tuples),len(self.doc_tuples)//10)},
        # })
        # <<<
    # <<<
