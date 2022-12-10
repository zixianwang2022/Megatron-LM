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
import json
import numpy as np
import os
import pickle
import time
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    FeatureExtractionPipeline,
)

from megatron import print_rank_0
from megatron.tokenizer import build_tokenizer


class HDF5Dataset(IterableDataset):
    def __init__(self, args, shared_dataset_info):
        self.indexed_datasets = shared_dataset_info["indexed_datasets"]
        self.dataset_ids = shared_dataset_info["dataset_ids"]
        self.chunk_index = shared_dataset_info["chunk_index"]
        self.tokenizer = build_tokenizer(args)

    def __iter__(self):
        for chunk_id in range(len(self.chunk_index)):

            doc_id, token_start_idx, token_end_idx, bert_chunk_len = \
                self.chunk_index[chunk_id]
            dataset_id = self.dataset_ids[chunk_id]
            chunk_len = token_end_idx - token_start_idx
            indexed_dataset = self.indexed_datasets[dataset_id]

            token_ids = indexed_dataset.get(doc_id,
                                            offset = token_start_idx,
                                            length = chunk_len)

            sample = self.tokenizer.detokenize(token_ids)
            text = sample.replace("<|endoftext|>", "")

            if len(sample) != len(text):
                # "chunk_id" : chunk_id,
                # "doc_id" : doc_id.item(),
                # "token_start_idx" : token_start_idx.item(),
                # "token_end_idx" : token_end_idx.item(),
                # "bert_chunk_len" : bert_chunk_len.item(),
                # "dataset_id" : dataset_id,
                # "token_ids" : token_ids,
                # "sample" : sample,
                # "text" : text,
                raise Exception("inconsistent.")

            yield text


class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def _forward(self, model_inputs):
        
        # >>>
        if 1:
            model_inputs.data["input_ids"][:] = 0
            model_inputs.data["input_ids"][0, 0] = 101
            model_inputs.data["input_ids"][0, 1] = 102
            # model_inputs.data[0, 2:] = 0
        # <<<

        model_outputs = self.model(**model_inputs)
        print("model input shape", model_inputs['attention_mask'].shape)
        embeddings = model_outputs[0]
        masks = torch.sum(model_inputs['attention_mask'], dim=1)

        # print(torch.mean(embeddings[0,1:-1], dim=0))

        outputs = []
        for embedding, mask in zip(embeddings, masks):
            outputs.append(torch.mean(embedding[1: mask - 1], dim=0))
        data = {
            "input" : model_inputs["input_ids"],
            "output" : outputs,
        }
        return data

    def postprocess(self, model_outputs):
        return {
            "input" : model_outputs["input"].numpy(),
            "output" : model_outputs["output"].numpy(),
        }


def test_huggingface(args, shared_dataset_info, timer):

    # Sort samples by bert chunk length.
    s = shared_dataset_info

    if 0:
        print_rank_0(" > sort / start.")
        t = time.time()
        bert_chunk_lens = list(enumerate(s["chunk_index"][:, 3]))
        bert_chunk_lens.sort(key = lambda item : item[1])
        print_rank_0(" > sort / end. [ %.2f sec ]" % (time.time() - t))
        sample_ids = [ item[0] for item in bert_chunk_lens ]
    else:
        sample_ids = list(range(len(s["chunk_index"])))

    offset = 0
    s["dataset_ids"] = np.array(s["dataset_ids"])
    s["dataset_ids"] = s["dataset_ids"][sample_ids[offset:]]
    s["chunk_index"] = s["chunk_index"][sample_ids[offset:]]

    model = BertModel.from_pretrained("bert-large-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-cased", model_max_length=256)

    # args = types.SimpleNamespace()
    args.device = 0
    args.bs = 1 # 1024
    args.split = 0 # overwrite.
    args.pointer = 0
    # args.input = ?
    # args.output = ?
    # <<<
    args.tokenizer_type = "GPT2BPETokenizer"
    args.vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json"
    args.merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt"
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    # dataset = HDF5Dataset(args.input, args)
    dataset = HDF5Dataset(args, shared_dataset_info)
    pipe = MyFeatureExtractionPipeline(
        model      = model,
        tokenizer  = tokenizer,
        device     = args.device,
        truncation = True,
        max_length = 256,
    )

    batch_size = args.bs

    # if os.path.exists(args.output):
    #     print("File already exits. Please check or delete it.")

    print(f"Streaming batch_size={batch_size}")


    n_chunks = len(shared_dataset_info["chunk_index"])
    # dset = np.zeros((len(dataset.data), 1024), dtype='float32')
    dset = np.zeros((n_chunks, 1024), dtype = "f4")
    print("data shape", dset.shape)
    pointer = 0
    # for out in tqdm(pipe(dataset, batch_size=batch_size), total=n_chunks):
    for idx, out_dict in enumerate(tqdm(pipe(dataset, batch_size=batch_size), total=n_chunks)):
        inp = out_dict["input"]
        out = out_dict["output"]
        if np.isnan(out).any():
            print("nan, %d, %d / %s." % (idx + offset, inp.size, str(inp)))
        dset[pointer] = out
        pointer += 1

    # f = h5py.File(args.output, 'w')
    # dset = f.create_dataset("feat", data=dset)
    # f.close()
