import json
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    FeatureExtractionPipeline,
)

# import argparse


# import sys
# sys.path.append("/home/boxinw-src/megatron-lm/megatron")
# sys.path.append("/home/boxinw-src/megatron-lm/")

# from megatron.tokenizer import build_tokenizer

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


class HDF5Dataset(IterableDataset):
    def __init__(self, file, args):
        self.file = h5py.File(file, "r")
        if 'chunks' in self.file:
            size = len(self.file['chunks'])
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
            self.data = np.reshape(self.data, (len(self.data) * 32, 64))  # chunk it

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

        # print(torch.mean(embeddings[0,1:-1], dim=0))

        outputs = []
        for embedding, mask in zip(embeddings, masks):
            outputs.append(torch.mean(embedding[1: mask - 1], dim=0))
        return {"embedding": outputs}

    def postprocess(self, model_outputs):
        return model_outputs["embedding"].numpy()


def test_huggingface(args, timer):

    # >>>
    from lutil import pax

    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    model = BertModel.from_pretrained("bert-large-cased")
    inputs = tokenizer(
        [
            # "hi there, i like your smile.",
            # "do the frown.",
            "",
        ],
        return_tensors = "pt",
        # padding = True,
        truncation = True,
    )
    inputs.data["input_ids"][0, :2] = 0
    # torch.from_numpy(np.array([ 101, 102 ])))
    outputs = model(**inputs)

    print(inputs.data["input_ids"])
    print(outputs["pooler_output"])
    pax({
        "model" : model,
        "inputs" : inputs,
        "outputs" : outputs,
        # "pool" : outputs["pooler_output"],
    })
    # <<<

    model = BertModel.from_pretrained("bert-large-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-cased", model_max_length=256)

    args = parser.parse_args()

    args.tokenizer_type = "GPT2BPETokenizer"
    args.vocab_file = "/home/boxinw-src/megatron-lm//gpt2-vocab.json"
    args.merge_file = "/home/boxinw-src/megatron-lm/gpt2-merges.txt"
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    dataset = HDF5Dataset(args.input, args)
    pipe = MyFeatureExtractionPipeline(model=model, tokenizer=tokenizer,
                                       device=args.device, truncation=True, max_length=256)

    batch_size = args.bs

    if os.path.exists(args.output):
        print("File already exits. Please check or delete it.")

    print(f"Streaming batch_size={batch_size}")


    dset = np.zeros((len(dataset.data), 1024), dtype='float32')
    print("data shape", dset.shape)
    pointer = 0
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset.data)):
        if np.isnan(out).any():
            out = np.nan_to_num(out, copy=False, nan=0)
        dset[pointer] = out
        pointer += 1

    f = h5py.File(args.output, 'w')
    dset = f.create_dataset("feat", data=dset)
    f.close()
