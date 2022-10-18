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


def get_dataset_map(args):

    # Load dataset metadata.
    with open(os.path.join(args.retro_workdir, "order.json")) as f:
        data_metas = json.load(f)

    # Token datasets.
    indexed_datasets = \
        [ make_indexed_dataset(m["prefix"], "mmap", True) for m in data_metas ]

    # Chunk index.
    chunk_index_path_map = get_chunk_index_path_map(args.retro_workdir)
    dataset_map = {}
    for key, chunk_index_path in chunk_index_path_map.items():

        # Load chunk index.
        f = h5py.File(chunk_index_path, "r")
        dataset_offsets = np.copy(f["dataset_offsets_valid"])
        chunk_index = np.copy(f["chunks_valid"])
        f.close()

        # Dataset ids.
        dataset_ids = []
        for i in range(len(dataset_offsets) - 1):
            dataset_ids.append([i] * (dataset_offsets[i+1] - dataset_offsets[i]))
        dataset_ids = [ i for ii in dataset_ids for i in ii ]

        # Dataset.
        dataset = BertChunkDataset(
            indexed_datasets = indexed_datasets,
            dataset_ids = dataset_ids,
            chunk_index = chunk_index,
            max_chunk_length = args.retro_chunk_length,
            max_model_seq_length = args.seq_length,
            masked_lm_prob = args.mask_prob,
            seed = args.seed,
        )

        dataset_map[key] = dataset

    return dataset_map


# def embed_chunks(args, timer):

#     # Embedding workdir.
#     workdir = os.path.join(args.retro_workdir, "embed")
#     os.makedirs(workdir, exist_ok = True)

#     # Load model.
#     models, optimizer, opt_param_scheduler = \
#         setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

#     # Dataset infos (indexed datasets, chunk index, etc.).
#     dataset_map = get_dataset_map(args)

#     # >>>
#     # del dataset_map["full"]
#     # <<<

#     # Embed each (i.e., full, sampled) dataset.
#     for prefix, dataset in dataset_map.items():
#         print_rank_0(" > embed '%s' chunks. [ count %d ]" %
#                      (prefix, len(dataset)))
#         embed_dataset_chunks(args, workdir, models, prefix, dataset)
# def embed_corpus_chunks(args, timer):
def embed_chunk_db(args, timer):

    raise Exception("call embed_text_datasets().")

    # Dataset infos (indexed datasets, chunk index, etc.).
    dataset_map = get_dataset_map(args)

    embed_text_datasets(texttttttttttttt)

