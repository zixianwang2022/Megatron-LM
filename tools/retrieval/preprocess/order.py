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

import glob
import h5py
# import joblib
import numpy as np
import os
from pathlib import Path
import time
import torch
from tqdm import tqdm

# import sys
# sys.path.append("/home/boxinw-src/megatron-lm/megatron")
# sys.path.append("/home/boxinw-src/megatron-lm/")

from megatron import get_args
# from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
# from megatron.tokenizer import build_tokenizer

# >>>
from lutil import pax
# <<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# see: notebook/faiss/create_chunks.ipynb
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# def dump_document_order():
def save_document_order():

    assert torch.distributed.get_rank() == 0, "single process operation."

    args = get_args()

    # pax(0, {
    #     "args" : args,
    #     "data_path" : args.data_path,
    # })

    def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
        """Build indexed dataset."""
        print(' > building dataset index ...')

        start_time = time.time()
        indexed_dataset = make_indexed_dataset(data_prefix,
                                               data_impl,
                                               skip_warmup)
        print(' > finished creating indexed dataset in {:4f} '
                     'seconds'.format(time.time() - start_time))
        print('    number of documents: {}'.format(
            indexed_dataset.sizes.shape[0]))

        return indexed_dataset

    data_files = [ prefix.rstrip("/") + ".bin" for prefix in args.data_path ]
    data_files = [ path for path in data_files if os.path.exists(path) ]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if 0:
        create_data_softlinks(data_files)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # pax(0, {"data_files": data_files})

    # for hdf in existing_chunk_files:
    #     f = h5py.File(hdf, "r")
    #     print(hdf, f["chunks"].shape, f["document_id"][-1])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 (393262503, 64) 190135
    # ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 (319135133, 64) 1189264
    # NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5 (5235460, 64) 740365
    # Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5 (40814306, 64) 26746
    # CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5 (1309682321, 64) 94208202
    # Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5 (66630804, 64) 5743989
    # rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5 (349458767, 64) 28198167
    # PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5 (74996373, 64) 14877028
    # CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5 (1088699591, 64) 77712318
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def get_database_and_index(indexed_dataset):

        size = indexed_dataset.sizes.shape[0]
        train = int(round(float(size) * 0.98))
        tot = 0

        databases = []
        indexes = []

        for document_id, document in enumerate(tqdm(indexed_dataset)):
            if document_id == train:
                break
            eod = document[-1]
            document = document[:-1]
            token_no = len(document)
            tot += token_no
            chunks = int(np.ceil(token_no / 64))

            for i in range(chunks):
                tokens = document[i * 64:(i+1) *64]
                if len(tokens) < 64:
                    pad = np.array([eod] * (64 - len(tokens)), dtype='uint16')
                    tokens = np.hstack((tokens, pad))
                assert len(tokens) == 64
                databases.append(tokens)
                indexes.append(document_id)
        return databases, indexes

    # for dataset in datasets[-6:]:
    for data_index, data_file in enumerate(data_files):

        data_name = Path(data_file).stem
        data_prefix = os.path.splitext(data_file)[0]
        chunk_file = data_prefix + ".chunks.hdf5"

        # pax(0, {
        #     "data_file" : data_file,
        #     "chunk_file" : chunk_file,
        # })

        if os.path.exists(chunk_file):
            continue

        # pax(0, {"data_name": data_name, "data_prefix": data_prefix})
        # print(data_name)

        print("creating chunks, dataset %d / %d ... '%s'." %
              (data_index, len(data_files), data_name))

        indexed_dataset = get_indexed_dataset_(
            data_prefix, # dataset[:-4],
            "mmap",
            True)
        databases, indexes = get_database_and_index(indexed_dataset)

        database = np.vstack(databases)
        index = np.array(indexes)

        f = h5py.File(chunk_file, "w")
        dset = f.create_dataset("chunks", data=database)
        dset = f.create_dataset("document_id", data=index)
        f.close()

        # pax(0, {
        #     "data_file" : data_file,
        #     "indexed_dataset" : type(indexed_dataset).__name__,
        #     "database" : type(database).__name__,
        #     "index" : type(index).__name__,
        # })

    # raise Exception("finished creating chunks.")

    #######################################################################
    #######################################################################
    #######################################################################

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # BookCorpus2_ftfy_cleaned_id_shuf_text_document
    # #  > building dataset index ...
    # #     reading sizes...
    # #     reading pointers...
    # #     reading document index...
    # #     creating numpy buffer of mmap...
    # #     creating memory view of numpy buffer...

    # #  > finished creating indexed dataset in 0.025186 seconds
    # #     number of documents: 18766

    # # stories_dedup0.7_shuf_cleaned_shuf_text_document
    # #  > building dataset index ...
    # #     reading sizes...
    # #     reading pointers...
    # #     reading document index...
    # #     creating numpy buffer of mmap...
    # #     creating memory view of numpy buffer...

    # #  > finished creating indexed dataset in 0.016499 seconds
    # #     number of documents: 670273
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # hdatasets = glob.glob("*.hdf5")

    # tot = 0
    # for dataset in hdatasets:
    #     f = h5py.File(dataset, "r")
    #     print(dataset, len(f['chunks']))
    #     tot += len(f['chunks'])

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 393262503
    # # Pile-CC_id_cleaned_shuf_text_document.chunks.hdf5 786507531
    # # ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 319135133
    # # OpenWebText2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 233910963
    # # NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5 5235460
    # # Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5 40814306
    # # CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5 1309682321
    # # Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5 66630804
    # # rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5 349458767
    # # PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5 74996373
    # # stories_dedup0.7_shuf_cleaned_shuf_text_document.chunks.hdf5 80830687
    # # Github_ftfy_id_shuf_text_document.chunks.hdf5 377166564
    # # BookCorpus2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 23578839
    # # StackExchange_ftfy_id_shuf_text_document.chunks.hdf5 184906924
    # # CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5 1088699591
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # print(tot)

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # 5334816766
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ARX="ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5"
    # BC2="BookCorpus2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5"
    # B3="Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5"
    # CC2020="CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5"
    # CC2021="CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5"
    # GIT="Github_ftfy_id_shuf_text_document.chunks.hdf5"
    # GUT="Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5"
    # NIH="NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5"
    # OWT2="OpenWebText2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5"
    # PCC="Pile-CC_id_cleaned_shuf_text_document.chunks.hdf5"
    # PM="PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5"
    # RN="rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5"
    # SE="StackExchange_ftfy_id_shuf_text_document.chunks.hdf5"
    # ST="stories_dedup0.7_shuf_cleaned_shuf_text_document.chunks.hdf5"
    # WIK="Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5"

    # DATA_BLEND={B3: 0.14336,
    #             RN: 0.08962,
    #             OWT2: 0.19336,
    #             SE: 0.05689,
    #             ST: 0.00859,
    #             PM: 0.02897,
    #             WIK: 0.04771,
    #             GUT: 0.00873,
    #             BC2: 0.01007,
    #             NIH:0.00208,
    #             CC2020: 0.13017,
    #             PCC:  0.09446,
    #             CC2021: 0.15652,
    #             ARX: 0.01359,
    #             GIT: 0.01588
    #            }

    # orders = [(k, v) for k, v in DATA_BLEND.items()]

    # f = h5py.File("pretraining_corpus" + ".chunks.hdf5", "w")
    # dset = f.create_dataset("chunks", (tot,64), dtype="uint16")

    # dset.shape

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # (5334816766, 64)
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # pointer = 0
    # for order in tqdm(orders):
    #     dataset = order[0]

    #     rf = h5py.File(dataset, "r")
    #     data = rf["chunks"]
    #     dset[pointer:pointer + len(data)] = data
    #     pointer += len(data)

    # f.close()

    # orders

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # [('Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.14336),
    # #  ('rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5', 0.08962),
    # #  ('OpenWebText2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.19336),
    # #  ('StackExchange_ftfy_id_shuf_text_document.chunks.hdf5', 0.05689),
    # #  ('stories_dedup0.7_shuf_cleaned_shuf_text_document.chunks.hdf5', 0.00859),
    # #  ('PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5', 0.02897),
    # #  ('Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5', 0.04771),
    # #  ('Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5',
    # #   0.00873),
    # #  ('BookCorpus2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.01007),
    # #  ('NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5', 0.00208),
    # #  ('CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5', 0.13017),
    # #  ('Pile-CC_id_cleaned_shuf_text_document.chunks.hdf5', 0.09446),
    # #  ('CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5', 0.15652),
    # #  ('ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.01359),
    # #  ('Github_ftfy_id_shuf_text_document.chunks.hdf5', 0.01588)]
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # joblib.dump(orders, "order.pkl")

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # ['order.pkl']
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # f = h5py.File("sampled_pretraining_corpus" + ".chunks.hdf5", "w")
    # sampled_tot = 300000000
    # dset = f.create_dataset("chunks", (sampled_tot,64), dtype="uint16")

    # pointer = 0
    # for order in tqdm(orders):
    #     dataset = order[0]
    #     ratio = order[1]
    #     size = int(round(float(sampled_tot) * ratio))

    #     rf = h5py.File(dataset, "r")
    #     data = rf["chunks"]
    #     dset[pointer:pointer + size] = data[:size]
    #     pointer += size

    # f.close()

    # f = h5py.File("pretraining_corpus" + ".chunks.hdf5", "r")

    # f['chunks'][2323453]

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # array([  547, 20467, 45427,    13,   632,   561,  1011,  4647,   284,
    # #        30282,   262,  3580,  1022,  3288,   290,  7593,  4808,  7645,
    # #           62, 27997,    13,  1892, 12362,    11,   262,  3288,  4808,
    # #         7645,    62, 27997,   287,  9215,  2900,   503,   284,   307,
    # #        13205,    11,  9472,   262,  7593,   318, 21499,  2728,  2279,
    # #          422,  4890,   284,  2612,  4369,   284, 47906, 15885,   198,
    # #          198,  1135,   783,   760,   326, 23426,   960,  8201,  5384,
    # #          960], dtype=uint16)
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # f['chunks'].shape

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # (5334816766, 64)
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # raise Exception("it worked?")

# eof
