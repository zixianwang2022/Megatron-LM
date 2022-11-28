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

import time

from megatron import print_rank_0
from tools.retro.utils import get_bert_tokenizer, get_gpt_tokenizer

from .utils import get_merged_dataset

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
