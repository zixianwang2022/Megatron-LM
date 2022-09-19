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

import h5py
import joblib
import numpy as np

# >>>
from lutil import pax, print_seq
# <<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# see: notebook/faiss/dump_offset_dict.ipynb
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def save_document_offsets():

    x = joblib.load('order.pkl')

    offset_dict = {}
    offset = 0 # ... i.e., document offset [ per dataset ]

    for data in x:
        offset_dict[data[0][:-26]] = offset
        print(data[0][:-26])
        with h5py.File(data[0], "r") as f:
            print(f['document_id'][-1x])
            offset += f['document_id'][-1]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Books3_ftfy_cleaned_id_shuf
    # 190135
    # rn_dedup_shuf_cleaned_0.7_cleaned_shuf
    # 28198167
    # OpenWebText2_ftfy_cleaned_id_shuf
    # 15361488
    # StackExchange_ftfy_id_shuf
    # 15011204
    # stories_dedup0.7_shuf_cleaned_shuf
    # 656867
    # PubMed_Abstracts_ftfy_id_shuf
    # 14877028
    # Wikipedia_en_ftfy_id_shuf
    # 5743989
    # Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf
    # 26746
    # BookCorpus2_ftfy_cleaned_id_shuf
    # 18390
    # NIH_ExPorter_ftfy_id_shuf
    # 740365
    # CC-2020-50_id_cleaned_shuf
    # 77712318
    # Pile-CC_id_cleaned_shuf
    # 49000874
    # CC-2021-04_id_cleaned_shuf
    # 94208202
    # ArXiv_ftfy_cleaned_id_shuf
    # 1189264
    # Github_ftfy_id_shuf
    # 10453153
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    offset

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 313388190
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    chunks = h5py.File('pretraining_corpus.chunks.hdf5', "a")
    len(chunks['chunks'])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 5334816766
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    chunks['chunks'].shape

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # (5334816766, 64)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    document_ids = np.zeros((5334816766), 'uint32')
    x

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # [('Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.14336),
    #  ('rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5', 0.08962),
    #  ('OpenWebText2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.19336),
    #  ('StackExchange_ftfy_id_shuf_text_document.chunks.hdf5', 0.05689),
    #  ('stories_dedup0.7_shuf_cleaned_shuf_text_document.chunks.hdf5', 0.00859),
    #  ('PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5', 0.02897),
    #  ('Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5', 0.04771),
    #  ('Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5',
    #   0.00873),
    #  ('BookCorpus2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.01007),
    #  ('NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5', 0.00208),
    #  ('CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5', 0.13017),
    #  ('Pile-CC_id_cleaned_shuf_text_document.chunks.hdf5', 0.09446),
    #  ('CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5', 0.15652),
    #  ('ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.01359),
    #  ('Github_ftfy_id_shuf_text_document.chunks.hdf5', 0.01588)]
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    from tqdm import tqdm

    pointer = 0

    for data in tqdm(x):
        print(data[0][:-26], offset_dict[data[0][:-26]])
        with h5py.File(data[0], "r") as f:
            size = len(f['document_id'])
            document_ids[pointer:pointer + size] = f['document_id']
            document_ids[pointer:pointer + size] += offset_dict[data[0][:-26]]
            pointer += size

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Books3_ftfy_cleaned_id_shuf 0
    # rn_dedup_shuf_cleaned_0.7_cleaned_shuf 190135
    # OpenWebText2_ftfy_cleaned_id_shuf 28388302
    # StackExchange_ftfy_id_shuf 43749790
    # stories_dedup0.7_shuf_cleaned_shuf 58760994
    # PubMed_Abstracts_ftfy_id_shuf 59417861
    # Wikipedia_en_ftfy_id_shuf 74294889
    # Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf 80038878
    # BookCorpus2_ftfy_cleaned_id_shuf 80065624
    # NIH_ExPorter_ftfy_id_shuf 80084014
    # CC-2020-50_id_cleaned_shuf 80824379
    # Pile-CC_id_cleaned_shuf 158536697
    # CC-2021-04_id_cleaned_shuf 207537571
    # ArXiv_ftfy_cleaned_id_shuf 301745773
    # Github_ftfy_id_shuf 302935037
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    offset_dict

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # {'Books3_ftfy_cleaned_id_shuf': 0,
    #  'rn_dedup_shuf_cleaned_0.7_cleaned_shuf': 190135,
    #  'OpenWebText2_ftfy_cleaned_id_shuf': 28388302,
    #  'StackExchange_ftfy_id_shuf': 43749790,
    #  'stories_dedup0.7_shuf_cleaned_shuf': 58760994,
    #  'PubMed_Abstracts_ftfy_id_shuf': 59417861,
    #  'Wikipedia_en_ftfy_id_shuf': 74294889,
    #  'Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf': 80038878,
    #  'BookCorpus2_ftfy_cleaned_id_shuf': 80065624,
    #  'NIH_ExPorter_ftfy_id_shuf': 80084014,
    #  'CC-2020-50_id_cleaned_shuf': 80824379,
    #  'Pile-CC_id_cleaned_shuf': 158536697,
    #  'CC-2021-04_id_cleaned_shuf': 207537571,
    #  'ArXiv_ftfy_cleaned_id_shuf': 301745773,
    #  'Github_ftfy_id_shuf': 302935037}
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    document_ids

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # array([        0,         0,         0, ..., 313388190, 313388190,
    #        313388190], dtype=uint32)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    chunks.create_dataset('document_id', data=document_ids)
    chunks.close()

    chunks = h5py.File('pretraining_corpus.chunks.hdf5', "r")

    chunks['document_id'][-1]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 313388190
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    joblib.dump(offset_dict, 'offset_dict.pkl')

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ['offset_dict.pkl']
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    all = np.ones((233, 5), 'int64')
    all[0] = [1,2,3,4]
    all

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # array([[1, 2, 3, 4, 5],
    #        [1, 1, 1, 1, 1],
    #        [1, 1, 1, 1, 1],
    #        ...,
    #        [1, 1, 1, 1, 1],
    #        [1, 1, 1, 1, 1],
    #        [1, 1, 1, 1, 1]])
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# eof
