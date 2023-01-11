# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import glob
import h5py
import numpy as np
import torch

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
