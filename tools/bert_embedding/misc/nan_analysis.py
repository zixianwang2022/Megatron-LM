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
import numpy as np
import os
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    FeatureExtractionPipeline,
)
from types import SimpleNamespace

from megatron.tokenizer import build_tokenizer


model = BertModel.from_pretrained("bert-large-cased")
tokenizer = AutoTokenizer.from_pretrained(
    "bert-large-cased",
    model_max_length = 256,
)


class HDF5Dataset(IterableDataset):
    def __init__(self, file, args):
        self.file = h5py.File(file, "r")
        if 'chunks' in self.file:
            data = self.file['chunks']
        else:
            data = self.file['tokens']
        size = len(data)
        print("data size", size)
        split_size = int(np.ceil(size / args.split))
        self.split_size = split_size
        print("split_size", split_size)
        start = split_size * args.pointer
        end = min(split_size * (args.pointer + 1), size)

        self.data = data[start:end]
        if 'tokens' in self.file:
            self.data = np.reshape(self.data, (len(self.data) * 32, 64)) #chunk it

        self.file.close()
        self.tokenizer = build_tokenizer(args)

    def __iter__(self):
        for data in self.data:
            sample = self.tokenizer.detokenize(data)
            text = sample.replace("<|endoftext|>", "")
            yield text


class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        print("model input shape", model_inputs['attention_mask'].shape)
        embeddings = model_outputs[0]
        masks = torch.sum(model_inputs['attention_mask'], dim=1)

        outputs = []
        for embedding, mask in zip(embeddings, masks):
            outputs.append(torch.mean(embedding[1: mask - 1], dim=0))
        return {"embedding": outputs}

    def postprocess(self, model_outputs):
        raise Exception("hi.")
        return model_outputs["embedding"].numpy()

def run_bert_nan_analysis(args, timer):

    if torch.distributed.get_rank() != 0:
        return

    # parser = argparse.ArgumentParser(description='Parse the input string.')
    # parser.add_argument('--device', type=int, default=0,
    #                     help='Device id to store the model')
    # parser.add_argument('--bs', type=int, default=1024,
    #                     help='batch size')
    # parser.add_argument('--split', type=int, default=0,
    #                     help='number of splits')
    # parser.add_argument('--pointer', type=int, default=0,
    #                     help='pointer id')
    # parser.add_argument('--input', type=str, default="",
    #                     help='Input to the jsonl file')
    # parser.add_argument('--output', type=str, default="",
    #                     help='Output of the extracted features')
    # args = parser.parse_args()

    i = 0
    start = i * 60000
    end = (i + 1) * 60000

    args = SimpleNamespace()
    args.device = 0
    # args.input = f"/lustre/fs1/projects/gpu_adlr/outputs/boxinw/chunks/pretraining_dump.h5py_start_{start}_end_{end}_ns_192000000_sl2048_seed_1234_with_offset.tokens.h5py"
    args.input = "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/sampled_pretraining_corpus.chunks.hdf5"
    # args.output = f"/lustre/fs1/projects/gpu_adlr/outputs/boxinw/chunks/pretraining_dump.h5py_start_{start}_end_{end}_ns_192000000_sl2048_seed_1234_with_offset.tokens.feat.hdf5"
    args.split = 1
    args.pointer = 0
    args.bs = 128

    args.tokenizer_type = "GPT2BPETokenizer"
    # args.vocab_file = "/home/boxinw-src/megatron-lm//gpt2-vocab.json"
    # args.merge_file = "/home/boxinw-src/megatron-lm/gpt2-merges.txt"
    args.vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/pile-cc1-cc2-shuf/bpe/gpt2-vocab.json"
    args.merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/pile-cc1-cc2-shuf/bpe/gpt2-merges.txt"
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    dataset = HDF5Dataset(args.input, args)

    pipe = MyFeatureExtractionPipeline(
        model = model,
        tokenizer = tokenizer,
        device = args.device,
        truncation = True,
        max_length = 256,
    )

    batch_size = args.bs

    # if os.path.exists(args.output):
    #     print("File already exits. Please check or delete it.")

    print(f"Streaming batch_size={batch_size}")

    dset = np.zeros((len(dataset.data), 1024), dtype='float32')
    print("data shape", dset.shape)
    pointer = 0
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset.data)):
        if np.isnan(out).any():
            out = np.nan_to_num(out, copy=False, nan=0)
        dset[pointer] = out
        pointer += 1

    # f = h5py.File(args.output, 'w')
    # dset = f.create_dataset("feat", data=dset)
    # f.close()
