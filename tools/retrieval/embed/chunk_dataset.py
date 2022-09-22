# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch

# >>>
from lutil import pax
# <<<


class GPTChunkDataset(torch.utils.data.Dataset):

    def __init__(self, indexed_dataset, chunk_index, eods):

        self.indexed_dataset = indexed_dataset
        self.chunk_index = chunk_index
        self.eods = eods

    def __len__(self):
        raise Exception("length?")
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):

        raise Exception("get item.")

        # >>>
        orig_idx = idx
        # <<<

        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]

        # >>>
        doc_ids = []
        # <<<
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f].item())
            # >>>
            # pax(0, {"data_prefix": self.data_prefix, "doc_ids": doc_ids})
            # <<<
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f].item())
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i].item())
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # >>>
            # pax(0, {"data_prefix": self.data_prefix, "doc_ids": doc_ids})
            # <<<
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        # >>>
        if self.return_doc_ids:
            if self.args.add_offset_doc_ids:
                doc_ids = [self.offset + x for x in doc_ids]
            data = {'text': np.array(sample, dtype=np.int64),
                    'doc_ids': doc_ids,
                    'idx': np.int32(orig_idx),
                    'dataset_ids': [self.data_prefix]}
            # pax(0, {"data": data})
            return data
        # <<<
        else:
            return {'text': np.array(sample, dtype=np.int64)}
