# lawrence mcafee

# import argparse
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

# import sys
# sys.path.append("/home/boxinw-src/megatron-lm/megatron")
# sys.path.append("/home/boxinw-src/megatron-lm/")

from megatron import print_rank_0
from megatron.tokenizer import build_tokenizer

# >>>
from lutil import pax, print_seq
# <<<


# class HDF5Dataset(IterableDataset):
#     def __init__(self, file, args):
#         self.file = h5py.File(file, "r")
#         if 'chunks' in self.file:
#             size = len(self.file['chunks'])
#             data = self.file['chunks']
#         else:
#             data = self.file['tokens']
#             size = len(data)
#         print("data size", size)
#         split_size = int(np.ceil(size / args.split))
#         self.split_size = split_size
#         print("split_size", split_size)
#         start = split_size * args.pointer
#         end = min(split_size * (args.pointer + 1), size)

#         self.data = data[start:end]
#         if 'tokens' in self.file:
#             self.data = np.reshape(self.data, (len(self.data) * 32, 64))  # chunk it

#         self.file.close()
#         self.tokenizer = build_tokenizer(args)

#     def __iter__(self):
#         for data in self.data:
#             sample = self.tokenizer.detokenize(data)
#             text = sample.replace("<|endoftext|>", "")
#             yield text
class HDF5Dataset(IterableDataset):
    def __init__(self, args, shared_dataset_info):
        self.indexed_datasets = shared_dataset_info["indexed_datasets"]
        self.dataset_ids = shared_dataset_info["dataset_ids"]
        self.chunk_index = shared_dataset_info["chunk_index"]
        self.tokenizer = build_tokenizer(args)

        # pax({"tokenizer": self.tokenizer})

        # >>>
        # # Sort samples by bert chunk length.
        # bert_chunk_lens = list(enumerate(self.chunk_index[:, 3]))
        # if 1:
        #     import time
        #     from megatron import print_rank_0

        #     print_rank_0(" > sort / start.")
        #     t = time.time()
        #     # >>> [ temporarily removed. ]
        #     bert_chunk_lens.sort(key = lambda item : item[1])
        #     # <<<
        #     # >>>
        #     # bert_chunk_lens.reverse() # for debugging.
        #     # <<<
        #     print_rank_0(" > sort / end. [ %.2f sec ]" % (time.time() - t))

        # self.sample_idxs = [ item[0] for item in bert_chunk_lens ]
        # # pax({"sample_idxs": self.sample_idxs})
        # <<<

    def __iter__(self):
        # for data in self.data:
        #     sample = self.tokenizer.detokenize(data)
        #     text = sample.replace("<|endoftext|>", "")
        #     yield text
        # for chunk_id, (doc_id,token_start_idx,token_end_idx,bert_chunk_len) in
        # for chunk_id in self.sample_idxs:
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
                pax({
                    "chunk_id" : chunk_id,
                    "doc_id" : doc_id.item(),
                    "token_start_idx" : token_start_idx.item(),
                    "token_end_idx" : token_end_idx.item(),
                    "bert_chunk_len" : bert_chunk_len.item(),
                    "dataset_id" : dataset_id,
                    "token_ids" : token_ids,
                    "sample" : sample,
                    "text" : text,
                })

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

        # pax({"model_inputs / data": model_inputs.data})
        
        outputs = []
        for embedding, mask in zip(embeddings, masks):
            outputs.append(torch.mean(embedding[1: mask - 1], dim=0))
        # return {"embedding": outputs}
        data = {
            "input" : model_inputs["input_ids"],
            "output" : outputs,
        }
        pax({"data": data})
        return data

    def postprocess(self, model_outputs):
        # pax({"model_outputs": model_outputs})
        # return model_outputs["embedding"].numpy()
        return {
            "input" : model_outputs["input"].numpy(),
            "output" : model_outputs["output"].numpy(),
        }


def test_huggingface(args, shared_dataset_info, timer):

    # >>>
    # from lutil import pax

    # tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    # model = BertModel.from_pretrained("bert-large-cased")
    # inputs = tokenizer(
    #     [
    #         # "hi there, i like your smile.",
    #         # "do the frown.",
    #         "",
    #     ],
    #     return_tensors = "pt",
    #     # padding = True,
    #     truncation = True,
    # )
    # inputs.data["input_ids"][0, :2] = 0
    # # torch.from_numpy(np.array([ 101, 102 ])))
    # outputs = model(**inputs)

    # print(inputs.data["input_ids"])
    # print(outputs["pooler_output"])
    # pax({
    #     "model" : model,
    #     "inputs" : inputs,
    #     "outputs" : outputs,
    #     # "pool" : outputs["pooler_output"],
    # })
    # <<<

    # >>>
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
    # offset = 84000
    # offset = 84587 # after nan 84587
    s["dataset_ids"] = np.array(s["dataset_ids"])
    s["dataset_ids"] = s["dataset_ids"][sample_ids[offset:]]
    s["chunk_index"] = s["chunk_index"][sample_ids[offset:]]

    # pax({
    #     "shared_dataset_info" : shared_dataset_info,
    #     "sample_ids" : str(sample_ids),
    # })
    # <<<

    # >>>
    if False:
        offset = 350000
        shared_dataset_info["dataset_ids"] = \
            shared_dataset_info["dataset_ids"][offset:]
        shared_dataset_info["chunk_index"] = \
            shared_dataset_info["chunk_index"][offset:]
        # pax({"shared_dataset_info": shared_dataset_info})
    # <<<

    model = BertModel.from_pretrained("bert-large-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-cased", model_max_length=256)

    # pax({"tokenizer": tokenizer})

    # >>>
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

    # pax({"args": args})

    # args = types.SimpleNamespace()
    args.device = 0
    args.bs = 1 # 1024
    args.split = 0 # overwrite.
    args.pointer = 0
    # args.input = ?
    # args.output = ?
    # <<<
    args.tokenizer_type = "GPT2BPETokenizer"
    # args.vocab_file = "/home/boxinw-src/megatron-lm//gpt2-vocab.json"
    # args.merge_file = "/home/boxinw-src/megatron-lm/gpt2-merges.txt"
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
        # >>>
        if 0:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            inp_str = str(inp)
            out_str = str(out)
            print("idx : %d [ %d ]" % (idx, idx + offset))
            print("inp : %d / %s" % (inp.size, inp_str))
            print("out : %d / %s ... %s" % (out.size,out_str[:20],out_str[-20:]))
        # <<<
        if np.isnan(out).any():
            # pax({
            #     "tokenizer" : tokenizer,
            #     "idx" : "%d [ %d ]" % (idx, idx + offset),
            #     "inp" : "%d / %s" % (inp.size, str(inp)),
            #     "out" : "%d / %s" % (out.size, str(out)),
            # })
            # raise Exception("nan.")
            # out = np.nan_to_num(out, copy=False, nan=0)
            print("nan, %d, %d / %s." % (idx + offset, inp.size, str(inp)))
        dset[pointer] = out
        pointer += 1

    # f = h5py.File(args.output, 'w')
    # dset = f.create_dataset("feat", data=dset)
    # f.close()
