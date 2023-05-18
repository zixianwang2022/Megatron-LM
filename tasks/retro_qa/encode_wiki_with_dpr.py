from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
import torch
import pandas as pd
import h5py
import json
import os
import pickle

import torch
from transformers import FeatureExtractionPipeline
from tqdm import tqdm
from torch.utils.data import IterableDataset
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse the input string.')
parser.add_argument('--device', type=int, default=0,
                    help='Device id to store the model')
parser.add_argument('--bs', type=int, default=128,
                    help='batch size')
parser.add_argument('--split', type=int, default=0,
                    help='number of splits')
parser.add_argument('--pointer', type=int, default=0,
                    help='pointer id')
parser.add_argument('--input', type=str, default="",
                    help='Input to the jsonl file')
parser.add_argument('--output', type=str, default="",
                    help='Output of the extracted features')
parser.add_argument('--dpr-mode', type=str, default="single",
                    help='Output of the extracted features')


def get_dpr_group(mode="single"):
    if mode == "single":
        dpr_path= "facebook/dpr-ctx_encoder-single-nq-base"
    elif mode == "multi":
        dpr_path= "facebook/dpr-ctx_encoder-multiset-base"
    else:
        raise ValueError("wrong mode for dpr")
    print(dpr_path)
    dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        dpr_path)
    dpr_model = DPRContextEncoder.from_pretrained(
        dpr_path).cuda()

    return (dpr_model, dpr_tokenizer)

def get_text_from_id(embed_ids, dpr_tokenizer):
    decode_text = dpr_tokenizer.decode(embed_ids)
    #decode_text = decode_text[len('[CLS] '): \
    #    len(decode_text)-len(' [SEP]')]
    return decode_text

def load_data(data_file):

    kilt_wiki_dataset = pd.read_json(data_file, lines=True)
    data = kilt_wiki_dataset['wiki_passage']
    return data

class WikiDataset(IterableDataset):

    def __init__(self, data_file, args):
        
        data = load_data(data_file)
        size = len(data)
        print("data size", size)
        split_size = int(np.ceil(size / args.split))
        self.split_size = split_size
        print("split_size", split_size)
        start = split_size * args.pointer
        end = min(split_size * (args.pointer + 1), size)

        self.data = data[start:end]

    def __iter__(self):
        for data in self.data:
            yield data

class DPRFeatureExtractionPipeline(FeatureExtractionPipeline):

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        # print("model input shape", model_inputs['attention_mask'].shape)
        outputs = model_outputs.pooler_output
        
        return {"embedding": outputs}

    def postprocess(self, model_outputs):
        return model_outputs["embedding"].numpy()


if __name__ == "__main__":
    
    args = parser.parse_args()
    dataset = WikiDataset(args.input, args)

    model, tokenizer = get_dpr_group(mode=args.dpr_mode)
    pipe = DPRFeatureExtractionPipeline(model=model, tokenizer=tokenizer,
                                   device=args.device, truncation=True, max_length=512)

    batch_size = args.bs

    if os.path.exists(args.output):
        print("File already exits. Please check or delete it.")

    print(f"Streaming batch_size={batch_size}")

    dset = np.zeros((len(dataset.data), 768), dtype='float32')
    print("data shape", dset.shape)
    pointer = 0
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset.data)):
        dset[pointer] = out
        pointer += 1

    f = h5py.File(args.output + "." + args.dpr_mode, 'w')
    dset = f.create_dataset("feat", data=dset)
    f.close()

