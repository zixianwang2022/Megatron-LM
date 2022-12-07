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

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    FeatureExtractionPipeline,
)

# >>>
from lutil import pax, print_seq
# <<<


class IterableTextDataset(torch.utils.data.IterableDataset):

    def __init__(self, text_dataset):
        self.text_dataset = text_dataset


    def __iter__(self):
        for sample_idx in range(len(self.text_dataset)):
            sample = self.text_dataset[sample_idx]
            text = sample["text"].replace("<|endoftext|>", "")
            # >>>
            # if "<|endoftext|>" in text:
            #     raise Exception("wha?")
            # <<<
            yield text


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
            # pax(0, {"output": output})
            # if torch.isnan(output).any():
            if torch.isnan(output.view(-1)[0]).any():
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

    def __init__(self, batch_size, max_seq_length):

        # >>> [ temporary, for debug. ]
        assert batch_size == 128
        assert max_seq_length == 256
        # <<<

        # model = BertModel.from_pretrained("bert-large-uncased")
        # tokenizer = AutoTokenizer.from_pretrained(
        #     "bert-large-uncased",
        #     model_max_lengthh = max_seq_length)
        self.model = BertModel.from_pretrained("bert-large-cased")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-cased", model_max_length=max_seq_length)

        self.pipe = MyFeatureExtractionPipeline(
            model      = self.model,
            tokenizer  = self.tokenizer,
            device     = torch.cuda.current_device(), # args.device,
            truncation = True,
            max_length = max_seq_length, # 256,
        )

        self.batch_size = batch_size


    def embed_text(self, text):
        
        class SingleTextDataset(torch.utils.data.Dataset):
            def __init__(self, text):
                assert isinstance(text, str)
                self.text = text
            def __len__(self):
                return 1
            def __getitem__(self, i):
                # assert i == 0
                return {"text": self.text}

        text_ds = SingleTextDataset(text)
        embed = self.embed_text_dataset(text_ds, verbose = False)[0]

        # token_ids = self.tokenizer(text)
        # embed = self.pipe._forward(token_ids)

        # pax(0, {
        #     "text" : text,
        #     "text_ds" : text_ds,
        #     "embed" : embed,
        # })

        return embed


    def embed_text_dataset(self, text_dataset, verbose = True):

        dataset = IterableTextDataset(text_dataset)

        n_samples = len(text_dataset)
        embeddings = np.zeros((n_samples, 1024), dtype = "f4")
        start_idx = 0
        # for idx, out_dict in enumerate(tqdm(
        #         self.pipe(dataset, batch_size = self.batch_size),
        #         total = n_samples,
        # )):
        _iter = self.pipe(dataset, batch_size = self.batch_size)
        if verbose:
            _iter = tqdm(_iter, total = n_samples)

        for idx, out_dict in enumerate(_iter):

            inp = out_dict["input"]
            out = out_dict["output"]
            # pax({"inp": inp, "out": out})
            embeddings[start_idx] = out
            start_idx += 1

        # pax(0, {"embeddings": embeddings})

        return embeddings
