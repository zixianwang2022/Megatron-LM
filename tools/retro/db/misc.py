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
import numpy as np
import time
import torch

from megatron import get_retro_args, print_rank_0
from tools.bert_embedding.huggingface import HuggingfaceEmbedder
from tools.retro.utils import get_bert_tokenizer, get_gpt_tokenizer

from .utils import get_merged_dataset, get_merged_train_dataset

# >>>
from lutil import pax
# <<<


# def print_longest_bert_chunks(args, shared_dataset_info):

#     n_chunks = len(shared_dataset_info["chunk_index"])

#     data_loader = get_block_data_loader(args,
#                                         shared_dataset_info,
#                                         0, n_chunks)
#     dataset = data_loader.dataset
#     gpt_tokenizer = dataset.gpt_tokenizer
#     bert_tokenizer = dataset.bert_tokenizer

#     # pax({"bert_tokenizer": bert_tokenizer})

#     print_rank_0(" > sort / start.")
#     t = time.time()
#     bert_chunk_lens = list(enumerate(dataset.chunk_index[:, 3]))
#     bert_chunk_lens.sort(key = lambda item : item[1])
#     bert_chunk_lens.reverse() # for debugging.
#     print_rank_0(" > sort / end. [ %.2f sec ]" % (time.time() - t))

#     results = []
#     for k, (chunk_id, bert_chunk_len) in enumerate(bert_chunk_lens[:20]):

#         gpt_token_ids = super(type(dataset),dataset).__getitem__(chunk_id)["text"]
#         text = gpt_tokenizer.detokenize(gpt_token_ids)
#         bert_token_ids = bert_tokenizer.tokenize(text)

#         print()
#         print()
#         print("#############################################################")
#         print("#############################################################")
#         print("#############################################################")
#         print("LENGTHS ... gpt %d, bert %d" % (len(gpt_token_ids), len(bert_token_ids)))
#         print("#############################################################")
#         print("GPT TOKENS ... %s" % ", ".join("(%d/%s)" % (i, str(gpt_tokenizer.detokenize([i])).replace("\n", "\\n").replace("\r", "\\r")) for i in gpt_token_ids))
#         print("#############################################################")
#         print("BERT TOKENS ... %s" % ", ".join("(%d/%s)" % (i, str(bert_tokenizer.inv_vocab[i]).replace("\n", "\\n").replace("\r", "\\r")) for i in bert_token_ids))
#         print("#############################################################")
        

#         # print("TEXT ... %s" % text)
#         print()
#         # print("####")
#         # print("####")
#         print(text)
#         print()

#         # pax({
#         #     "text" : text,
#         #     "gpt_token_ids" : "%d / %s" % (
#         #         len(gpt_token_ids),
#         #         str(gpt_token_ids.tolist()),
#         #     ),
#         #     "bert_token_ids" : "%d / %s" % (
#         #         len(bert_token_ids),
#         #         str(bert_token_ids),
#         #     ),
#         # })
        
#     torch.distributed.barrier()
#     exit(0)

#     pax(0, {
#         "shared_dataset_info" : shared_dataset_info,
#         "n_chunks" : n_chunks,
#         "data_loader" : data_loader,
#         "bert_chunk_lens" : bert_chunk_lens[:10],
#     })
def print_longest_bert_chunks():

    # dataset = get_merged_dataset("train")
    dataset = get_individual_dataset("wiki")
    n_chunks = len(dataset)

    gpt_tokenizer = get_gpt_tokenizer()
    bert_tokenizer = get_bert_tokenizer()

    # pax(0, {
    #     "n_chunks" : n_chunks,
    #     "gpt_tokenizer" : gpt_tokenizer,
    #     "bert_tokenizer" : bert_tokenizer,
    # })

    print_rank_0(" > sort / start.")
    t = time.time()
    bert_chunk_lens = list(enumerate(dataset.chunk_db[:, 3]))
    raise Exception("hi.")
    bert_chunk_lens.sort(key = lambda item : item[1])
    bert_chunk_lens.reverse() # for debugging.
    print_rank_0(" > sort / end. [ %.2f sec ]" % (time.time() - t))

    results = []
    for k, (chunk_id, bert_chunk_len) in enumerate(bert_chunk_lens[:20]):

        gpt_token_ids = super(type(dataset),dataset).__getitem__(chunk_id)["text"]
        text = gpt_tokenizer.detokenize(gpt_token_ids)
        bert_token_ids = bert_tokenizer.tokenize(text)

        print()
        print()
        print("#############################################################")
        print("#############################################################")
        print("#############################################################")
        print("LENGTHS ... gpt %d, bert %d" % (len(gpt_token_ids), len(bert_token_ids)))
        print("#############################################################")
        print("GPT TOKENS ... %s" % ", ".join("(%d/%s)" % (i, str(gpt_tokenizer.detokenize([i])).replace("\n", "\\n").replace("\r", "\\r")) for i in gpt_token_ids))
        print("#############################################################")
        print("BERT TOKENS ... %s" % ", ".join("(%d/%s)" % (i, str(bert_tokenizer.inv_vocab[i]).replace("\n", "\\n").replace("\r", "\\r")) for i in bert_token_ids))
        print("#############################################################")
        

        # print("TEXT ... %s" % text)
        print()
        # print("####")
        # print("####")
        print(text)
        print()

        # pax({
        #     "text" : text,
        #     "gpt_token_ids" : "%d / %s" % (
        #         len(gpt_token_ids),
        #         str(gpt_token_ids.tolist()),
        #     ),
        #     "bert_token_ids" : "%d / %s" % (
        #         len(bert_token_ids),
        #         str(bert_token_ids),
        #     ),
        # })
        
    torch.distributed.barrier()
    exit(0)

    pax(0, {
        "shared_dataset_info" : shared_dataset_info,
        "n_chunks" : n_chunks,
        "data_loader" : data_loader,
        "bert_chunk_lens" : bert_chunk_lens[:10],
    })


class OldEmbedDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()

        chunk_path = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5"
        self.chunks = h5py.File(chunk_path)["chunks"]

        self.embed_paths = sorted(glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/wiki-cls-indexes/*.hdf5"))
        self.embed_offsets = [0]
        for p in self.embed_paths:
            with h5py.File(p) as f:
                self.embed_offsets.append(self.embed_offsets[-1] + len(f["feat"]))

        # pax(0, {
        #     "chunks / shape" : str(self.chunks.shape),
        #     # "embed_paths" : embed_paths,
        #     # "embed_sizes" : embed_sizes,
        #     # "embed_sizes / total" : sum(embed_sizes),
        #     "embed_offsets" : str(self.embed_offsets),
        # })

    def __getitem__(self, idx):

        tokens = np.copy(self.chunks[idx])

        for i in range(len(self.embed_offsets) - 1):
            start_idx = self.embed_offsets[i]
            end_idx = self.embed_offsets[i + 1]
            if idx >= start_idx and idx < end_idx:
                embed_path = self.embed_paths[i]
                break

        with h5py.File(embed_path) as f:
            embed = np.copy(f["feat"][idx - start_idx])

        # pax(0, {
        #     "idx" : idx,
        #     "tokens" : tokens,
        #     "embed_path" : embed_path,
        #     "embed" : embed,
        # })

        return {
            "tokens" : tokens,
            "embed" : embed,
        }


class NewEmbedDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()

        from tools.retro.pretraining.retro_dataset import get_chunk_path_map
        args = get_retro_args()

        self.block_size = args.retro_block_size
        self.chunk_ds = get_merged_train_dataset()
        self.embed_path_map = get_chunk_path_map("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/training_data_tmp/blocks")


    def __getitem__(self, idx):

        embed_path = self.embed_path_map[idx]
        with h5py.File(embed_path, "r") as f:
            embed = np.copy(f["data"][idx % self.block_size])

        # if idx > 0:
        #     pax(0, {
        #         "idx" : idx,
        #         "embed_path" : embed_path,
        #         "embed" : embed,
        #     })

        return {
            "tokens" : self.chunk_ds[idx]["text"],
            "embed" : embed,
        }


def print_db_embeddings():

    args = get_retro_args()
    embedder = HuggingfaceEmbedder(128, 256)
    gpt_tokenizer = get_gpt_tokenizer()

    old_ds = OldEmbedDataset()
    new_ds = NewEmbedDataset()
    
    from tools.retro.pretraining.misc import align_db_idxs
    old_db_hash_map, new_db_hash_map, common_db_hashes = align_db_idxs(
        old_ds.chunks,
        new_ds.chunk_ds,
    )

    common_db_hashes = list(common_db_hashes)
    np.random.shuffle(common_db_hashes)

    accs = []
    n_common = len(common_db_hashes)
    # for db_hash_idx in range(0, n_common, n_common // 100):
    #     db_hash = common_db_hashes[db_hash_idx]
    # for db_hash_idx in range(10):
    #     old_id = db_hash_idx
    #     new_id = db_hash_idx
    sampled_db_hashes = common_db_hashes[0:n_common:(n_common//1000)]
    for db_hash in sampled_db_hashes:

        old_id = old_db_hash_map[db_hash]
        new_id = new_db_hash_map[db_hash]

        old_sample = old_ds[old_id]
        new_sample = new_ds[new_id]

        assert np.array_equal(old_sample["tokens"], new_sample["tokens"])
        tokens = old_sample["tokens"]
        text = gpt_tokenizer.detokenize(tokens)
        old_embed = old_sample["embed"].flatten()
        new_embed = new_sample["embed"].flatten()

        if np.array_equal(old_embed, new_embed):
            accs.append(1)
            continue
        accs.append(0)

        hf_embed = embedder.embed_text(text).flatten()

        header = "############## chunk %s ##############" % (
            ",".join(str(i) for i in set([old_id, new_id])))
        print()
        print("#" * len(header))
        print(header)
        print("#" * len(header))
        print("CHUNK   : %s" % "\\n".join(text.splitlines()))
        print("OLD_EMB : %s ..." % str(old_embed.tolist())[:125])
        print("NEW_EMB : %s ..." % str(new_embed.tolist())[:125])
        print("HF_EMB  : %s ..." % str(hf_embed.tolist())[:125])
        print("EQUAL?  : %d." % np.array_equal(old_embed, new_embed))

        # pax(0, {
        #     "hash" : hash,
        #     "old_sample" : old_sample,
        #     "new_sample" : new_sample,
        #     "equal?" : np.array_equal(old_sample["embed"], new_sample["embed"]),
        # })

    print("acc = %.2f [ n %d ]." % (100 * np.mean(accs), len(accs)))
    exit()
