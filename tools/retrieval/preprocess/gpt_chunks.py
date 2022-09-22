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

import os
import torch

# from megatron import get_args
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.training import (
    build_train_valid_test_data_iterators,
    update_train_iters,
)
from pretrain_gpt import train_valid_test_datasets_provider

# >>>
from lutil import pax
# <<<


def build_gpt_chunk_index(args, timer):

    assert torch.distributed.get_rank() == 0, "single process operation."

    # args = get_args()

    # pax(0, {
    #     "args" : args,
    #     "data_path" : args.data_path,
    # })

    # >>>
    # args.iteration = 0
    # update_train_iters(args)
    # pax({"args": args})
    # train_data_iterator, valid_data_iterator, test_data_iterator \
    #     = build_train_valid_test_data_iterators(
    #         train_valid_test_datasets_provider)
    # <<<

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))

    pax({
        "train_ds" : train_ds,
    })
    # <<<

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

    pax(0, {"data_files": data_files})

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
