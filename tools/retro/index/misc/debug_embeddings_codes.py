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

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding import BertEmbedder, DiskDataParallelBertEmbedder
from tools.retro.db.utils import (
    get_merged_train_dataset,
    get_merged_sampled_dataset,
)
from tools.retro.utils import GPTToTextDataset

# >>>
from lutil import pax
# <<<


# >>> [ fake add! ]
# def add(self, text_dataset):

#     # pax(0, {"n_threads" : faiss.omp_get_max_threads()})

#     if torch.distributed.get_rank() != 0:
#         return

#     import glob

#     added_index_path = self.get_added_index_path()
#     if os.path.exists(added_index_path):
#         raise Exception("index already written.")

#     embedding_paths = sorted(glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/train_tmp/*.hdf5"))
#     code_paths = sorted(glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/add-v0/add_tmp/*.hdf5"))

#     assert len(embedding_paths) == len(code_paths)

#     # pax(0, {
#     #     "embedding_paths" : embedding_paths,
#     #     "code_paths" : code_paths,
#     # })

#     from tools.bert_embedding import BertEmbedder
#     from tools.retro.db.utils import get_merged_valid_dataset
#     from tools.retro.utils import GPTToTextDataset
#     args = get_retro_args()
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("embedder.")
#     # embedder = BertEmbedder(
#     #     args.retro_bert_batch_size,
#     #     args.retro_bert_max_chunk_length,
#     #     "megatron",
#     # )
#     embedder = BertEmbedder(args.retro_bert_batch_size,
#                             args.retro_bert_max_chunk_length,
#                             args.bert_embedder_type)
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("valid dataset.")
#     valid_dataset = GPTToTextDataset(get_merged_valid_dataset())
#     print("valid text.")
#     valid_text = valid_dataset[0]["text"]
#     print("valid embed.")
#     valid_embed = embedder.embed_text(valid_text).reshape((1, -1))

#     # pax({
#     #     "valid_text" : valid_text,
#     #     "valid_embed" : valid_embed,
#     # })

#     index0 = self.get_empty_index()
#     index1 = faiss.clone_index(index0)

#     for index in (index0, index1):
#         faiss.ParameterSpace().set_index_parameter(index, "efSearch",
#                                                    args.retro_ef_search)
#         faiss.ParameterSpace().set_index_parameter(index, "nprobe",
#                                                    args.retro_nprobe)


#     # Add embeddings.
#     print("add embeddings.")
#     # for path_idx, embedding_path in enumerate(tqdm(embedding_paths)):
#     # for path_idx in tqdm(range(66, len(embedding_paths))):
#     for path_idx in tqdm(range(len(embedding_paths))):

#         with h5py.File(embedding_paths[path_idx]) as f:
#             embeddings0 = np.copy(f["data"])
#             index0.add(embeddings0)
#         with h5py.File(code_paths[path_idx]) as f:
#             codes1 = np.copy(f["data"])
#             index1.add_sa_codes(codes1)

#         bal0 = index0.invlists.imbalance_factor()
#         bal1 = index1.invlists.imbalance_factor()

#         nbrs0 = index0.search(valid_embed, 5)
#         nbrs1 = index1.search(valid_embed, 5)

#         if True or bal0 != bal1 or not np.array_equal(nbrs0, nbrs1):
#             codes0 = index0.sa_encode(embeddings0)
#             codes0_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/scratch/codes0.hdf5"
#             with h5py.File(codes0_path, "w") as f:
#                 f.create_dataset("data", data = codes0)
#             with h5py.File(codes0_path) as f:
#                 codes0b = np.copy(f["data"])
#             pax(0, {
#                 # "n_threads" : faiss.omp_get_max_threads(),
#                 "path_idx" : path_idx,
#                 "index0 / ntotal" : index0.ntotal,
#                 "index1 / ntotal" : index1.ntotal,
#                 "bal0" : bal0,
#                 "bal1" : bal1,
#                 "nbrs0" : nbrs0,
#                 "nbrs1" : nbrs1,
#                 "--" : "--",
#                 "embeddings0" : str(embeddings0.flatten().tolist()),
#                 "codes0" : str(codes0.flatten().tolist()),
#                 "codes0b" : str(codes0b.flatten().tolist()),
#                 "codes1" : str(codes1.flatten().tolist()),
#                 "codes0 == codes1" : np.array_equal(codes0, codes1),
#                 "codes0 == codes0b" : np.array_equal(codes0, codes0b),
#             })

#         # break

#     # Write index.
#     print_rank_0("write added index.")
#     faiss.write_index(index0, added_index_path)

#     raise Exception("index written.")
def add(self, text_dataset):

    if torch.distributed.get_rank() != 0:
        return

    args = get_retro_args()

    # Get text dataset.
    sampled_dataset = GPTToTextDataset(get_merged_sampled_dataset())
    train_dataset = GPTToTextDataset(get_merged_train_dataset())

    # Index.
    index = self.get_empty_index()

    # Embedder
    embedder = BertEmbedder(args.retro_bert_batch_size,
                            args.retro_bert_max_chunk_length,
                            args.bert_embedder_type)

    # >>>
    # # embs0 = [ embedder.embed_text("lawrence the great.") for _ in range(5) ]
    # # embs1 = [ embedder.embed_text("the life.") for _ in range(5) ]
    # embs0 = [ embedder.embed_text_dataset(torch.utils.data.Subset(sampled_dataset, range(100))) for _ in range(5) ]
    # pax({
    #     "embs0" : embs0,
    #     # "embs1" : embs1,
    # })
    # <<<

    # >>>
    embs0 = embedder.embed_text_dataset(torch.utils.data.Subset(sampled_dataset, range(args.retro_block_size)))
    pax({"embs0": embs0})
    # <<<

    # Embedding, code paths.
    embedding_paths = sorted(glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/train_tmp/*.hdf5"))
    code_paths = sorted(glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/add-v0/add_tmp/*.hdf5"))
    assert len(embedding_paths) == len(code_paths)

    # Compare codes.
    print("compare codes.")
    # for path_idx, embedding_path in enumerate(tqdm(embedding_paths)):
    # for path_idx in tqdm(range(66, len(embedding_paths))):
    for path_idx in tqdm(range(len(embedding_paths))):
    # for path_idx in tqdm(range(64, len(embedding_paths))):

        embedding_path = embedding_paths[path_idx]
        code_path = code_paths[path_idx]

        with h5py.File(embedding_path) as f:
            embeddings0 = np.copy(f["data"])
            codes0 = index.sa_encode(embeddings0)
        with h5py.File(code_path) as f:
            codes1 = np.copy(f["data"])

        if True or not np.array_equal(codes0, codes1):

            # >>>
            def get_block(prefix):
                block_range = (
                    path_idx * args.retro_block_size,
                    (path_idx + 1) * args.retro_block_size,
                )
                return {
                    "range" : block_range,
                    "path" : "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/scratch/%s-%d-%d.hdf5" % (prefix, *block_range),
                }

            # pax(0, {"block": block})
            print_rank_0("> encode block / sampled.")
            s_embeddings = self.encode_block(index, embedder, sampled_dataset, get_block("s"))
            print_rank_0("> encode block / train.")
            t_embeddings = self.encode_block(index, embedder, train_dataset, get_block("t"))

            pax(0, {
                "path_idx" : path_idx,
                "embedding_path" : embedding_path,
                "embeddings0" : str(embeddings0.flatten().tolist()),
                "s_embeddings" : str(s_embeddings.flatten().tolist()),
                "t_embeddings" : str(t_embeddings.flatten().tolist()),
            })

            print_rank_0("load codes1b.")
            with h5py.File(block["path"]) as f:
                codes1b = np.copy(f["data"])
            # <<<

            pax(0, {
                "path_idx" : path_idx,
                "embedding_path" : embedding_path,
                "code_path" : code_path,
                "embeddings0" : str(embeddings0.flatten().tolist()),
                "embeddings1b" : str(embeddings1b.flatten().tolist()),
                "codes0" : str(codes0.flatten().tolist()),
                "codes1" : str(codes1.flatten().tolist()),
                "codes1b" : str(codes1b.flatten().tolist()),
                "codes0 == codes1" : np.array_equal(codes0, codes1),
            })

    raise Exception("fin.")


train_data_dir = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/scratch/train_tmp"

def embed_db_mockup():
    '''Embed DB chunks.

    Store chunks in blocks on disk. These blocks will later be merged into
    a single dataset for training the index.
    '''

    # # Embed only if index not already trained.
    # empty_index_path = get_empty_index_path()
    # if os.path.isfile(empty_index_path):
    #     raise Exception("unreachable.")
    #     return

    args = get_retro_args()

    # Get db dataset.
    gpt_dataset = get_merged_sampled_dataset()
    text_dataset = GPTToTextDataset(gpt_dataset)

    # Embed dataset.
    embedder = DiskDataParallelBertEmbedder(args.retro_bert_batch_size,
                                            args.retro_bert_max_chunk_length,
                                            args.retro_block_size,
                                            args.bert_embedder_type)
    embedder.embed_text_dataset("index", train_data_dir, text_dataset)

import glob
import h5py
import numpy as np
import torch

def print_training_embeddings():

    if torch.distributed.get_rank() != 0:
        return

    embedding_paths = sorted(glob.glob(train_data_dir + "/*hdf5"))
    embeddings = []
    for embedding_path in embedding_paths:
        with h5py.File(embedding_path) as f:
            embeddings.append(np.copy(f["data"]))
    pax(0, {"embeddings": embeddings})

    pax({"embedding_paths": embedding_paths})

def compare_interactive_batch():

    if torch.distributed.get_rank() != 0:
        return

    interactive_paths = sorted(glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/train_tmp_interactive/*.hdf5"))
    batch_paths = sorted(glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/train_tmp_batch/*.hdf5"))

    # pax({
    #     "interactive_paths" : interactive_paths,
    #     "batch_paths" : batch_paths,
    # })

    for ipath, bpath in zip(interactive_paths, batch_paths):
        with h5py.File(ipath) as f:
            iembeds = np.copy(f["data"])
        with h5py.File(bpath) as f:
            bembeds = np.copy(f["data"])
        pax({
            "ipath" : ipath,
            "bpath" : bpath,
            "iembeds" : iembeds,
            "bembeds" : bembeds,
        })

    embeddings = []
    for embedding_path in embedding_paths:
        with h5py.File(embedding_path) as f:
            embeddings.append(np.copy(f["data"]))
    pax(0, {"embeddings": embeddings})

    pax({"embedding_paths": embedding_paths})

def test_bert_load():

    args = get_retro_args()
    embedder = BertEmbedder(
        args.retro_bert_batch_size,
        args.retro_bert_max_chunk_length,
        args.bert_embedder_type,
    )

    if torch.distributed.get_rank() != 0:
        return

    text_dataset = GPTToTextDataset(get_merged_train_dataset())
    embeddings = embedder.embed_text_dataset(torch.utils.data.Subset(text_dataset, range(3)))

    pax({"embeddings": str(embeddings.flatten().tolist())})

def debug_embeddings_codes():

    # embed_db_mockup()
    # print_training_embeddings()
    # compare_interactive_batch()
    test_bert_load()

    torch.distributed.barrier()
    exit()
