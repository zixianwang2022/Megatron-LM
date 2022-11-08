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

# import h5py
# import json
import numpy as np
# import os
# import pickle
# import time
import torch
# from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    # BertTokenizer,
    FeatureExtractionPipeline,
)

# from megatron import print_rank_0
# from megatron.tokenizer import build_tokenizer

# >>>
from lutil import pax, print_seq
# <<<


# class HDF5Dataset(IterableDataset):
#     def __init__(self, args, shared_dataset_info):
#         self.indexed_datasets = shared_dataset_info["indexed_datasets"]
#         self.dataset_ids = shared_dataset_info["dataset_ids"]
#         self.chunk_index = shared_dataset_info["chunk_index"]
#         self.tokenizer = build_tokenizer(args)

#     def __iter__(self):
#         for chunk_id in range(len(self.chunk_index)):

#             doc_id, token_start_idx, token_end_idx, bert_chunk_len = \
#                 self.chunk_index[chunk_id]
#             dataset_id = self.dataset_ids[chunk_id]
#             chunk_len = token_end_idx - token_start_idx
#             indexed_dataset = self.indexed_datasets[dataset_id]

#             token_ids = indexed_dataset.get(doc_id,
#                                             offset = token_start_idx,
#                                             length = chunk_len)

#             sample = self.tokenizer.detokenize(token_ids)
#             text = sample.replace("<|endoftext|>", "")

#             if len(sample) != len(text):
#                 pax({
#                     "chunk_id" : chunk_id,
#                     "doc_id" : doc_id.item(),
#                     "token_start_idx" : token_start_idx.item(),
#                     "token_end_idx" : token_end_idx.item(),
#                     "bert_chunk_len" : bert_chunk_len.item(),
#                     "dataset_id" : dataset_id,
#                     "token_ids" : token_ids,
#                     "sample" : sample,
#                     "text" : text,
#                 })

#             yield text
# class TextDatasetWrapper(torch.utils.data.IterableDataset):
class IterableTextDataset(torch.utils.data.IterableDataset):

    def __init__(self, text_dataset):
        self.text_dataset = text_dataset


    def __iter__(self):
        for sample_idx in range(len(self.text_dataset)):
            sample = self.text_dataset[sample_idx]
            yield sample["text"]


# class HuggingfaceEmbedder(FeatureExtractionPipeline):
class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def _forward(self, model_inputs):

        # >>>
        # if 1:
        #     model_inputs.data["input_ids"][:] = 0
        #     model_inputs.data["input_ids"][0, 0] = 101
        #     model_inputs.data["input_ids"][0, 1] = 102
        #     # model_inputs.data[0, 2:] = 0
        # <<<

        model_outputs = self.model(**model_inputs)

        # pax(0, {
        #     "model_inputs" : model_inputs,
        #     "model_outputs" : model_outputs,
        # })
        
        embeddings = model_outputs[0]
        masks = torch.sum(model_inputs['attention_mask'], dim=1)

        outputs = []
        for embedding, mask in zip(embeddings, masks):
            output = torch.mean(embedding[1: mask - 1], dim=0)
            if torch.isnan(output).any():
                # raise Exception("here we go.")
                output.zero_()
            outputs.append(output)
        data = {
            "input" : model_inputs["input_ids"],
            "output" : outputs,
            # "output" : torch.stack(outputs),
        }
        # pax({"data": data, "n_outputs": len(data["output"])})
        return data

    def postprocess(self, model_outputs):
        # pax({"model_outputs": model_outputs})
        # return model_outputs["embedding"].numpy()
        return {
            "input" : model_outputs["input"].numpy(),
            "output" : model_outputs["output"].numpy(),
        }


class HuggingfaceEmbedder:

    def __init__(self, max_seq_length, batch_size):

        model = BertModel.from_pretrained("bert-large-uncased")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased",
            # model_max_length=256)
            model_max_lengthh = max_seq_length)

        # pax(0, {
        #     "model" : self.model,
        #     "tokenizer" : self.tokenizer,
        #     "device" : torch.cuda.current_device(),
        # })
        
        # # args = types.SimpleNamespace()
        # args.device = torch.distributed.get_rank()
        # args.bs = 1 # 1024
        # args.split = 0 # overwrite.
        # args.pointer = 0
        # # args.input = ?
        # # args.output = ?
        # # <<<
        # args.tokenizer_type = "GPT2BPETokenizer"
        # # args.vocab_file = "/home/boxinw-src/megatron-lm//gpt2-vocab.json"
        # # args.merge_file = "/home/boxinw-src/megatron-lm/gpt2-merges.txt"
        # args.vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json"
        # args.merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt"
        # args.rank = 0
        # args.make_vocab_size_divisible_by = 128
        # args.tensor_model_parallel_size = 1
        # args.vocab_extra_ids = 0

        # dataset = HDF5Dataset(args.input, args)
        # dataset = HDF5Dataset(args, shared_dataset_info)
        self.pipe = MyFeatureExtractionPipeline(
            model      = model,
            tokenizer  = tokenizer,
            device     = torch.cuda.current_device(), # args.device,
            truncation = True,
            max_length = max_seq_length, # 256,
        )

        # batch_size = args.bs
        # self.batch_size = 128
        self.batch_size = batch_size


    def embed_text_dataset(self, text_dataset):

        dataset = IterableTextDataset(text_dataset)

        n_samples = len(text_dataset)
        embeddings = np.zeros((n_samples, 1024), dtype = "f4")
        # >>>
        # return embeddings
        # <<<
        start_idx = 0
        for idx, out_dict in enumerate(tqdm(
                self.pipe(dataset, batch_size = self.batch_size),
                total = n_samples,
        )):

            inp = out_dict["input"]
            out = out_dict["output"]
            # pax({"inp": inp, "out": out})
            embeddings[start_idx] = out
            start_idx += 1

        # pax(0, {"embeddings": embeddings})

        return embeddings
