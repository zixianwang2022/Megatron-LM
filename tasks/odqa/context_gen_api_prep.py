# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Prompting the pretrained language model to generate knowledge/response"""

import json
import requests
import random
import os.path
import time
import argparse
from email.utils import encode_rfc2231
import json
import torch
import requests
from nltk import word_tokenize
from megatron import mpu
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.text_generation import generate_and_post_process, beam_search_and_post_process
from .data import load_data, load_data_distributed, load_piQA_data, load_data_qg, load_data_kilt
from .retriever import MyRetriever
from .utils import write_output
import random
import os.path
from pathlib import Path
import shutil
import time
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder


def prompt_sample_selection(data_list, query = "", k=10, is_random=True, retriever=None):

    args = get_args()

    if k==0:
        return []

    if is_random:
        print("random select the top-k samples")
        
        return random.sample(data_list, k)
    else: 
        ## option1: return the top-k
        assert retriever is not None
        # print("select the samples based on query : {} similarity!".format(args.emb_type))
        if args.remove_duplicate_ctx:
            return retriever.get_topk(query, k+10, args.emb_type)
        else:
            return retriever.get_topk(query, k, args.emb_type)


def context_construction_for_api_call(input_list, list_of_topk_list, num_prompt_examples=0,):

    args = get_args()

    megatron_tokenizer =  get_tokenizer()
    megatron_tokenizer.pad_token = "<|endoftext|>"

    assert megatron_tokenizer.tokenize('hello\n\nhello') == [31373, 198, 198, 31373]

    # return the generated context
    context_prompt_list = []
    context_prompt_len_list=[]
    generation_list = []

    if args.save_context_path is not None and os.path.exists(args.save_context_path):
        print("loading the context_gen_file from {}".format(args.save_context_path))
        with open(args.save_context_path, 'r') as f:
            ctx_generation_list = f.readlines()
        for input in input_list:
            generation_list.append(ctx_generation_list[int(input['id'])])
        return generation_list

    else:
        for input, topk_list in zip(input_list,list_of_topk_list):
            context_prompt = construct_context_prompt(input, topk_list, num_prompt_examples, args.remove_duplicate_ctx, tokenizer=megatron_tokenizer)
            context_prompt_len_list.append(len(context_prompt))
            context_prompt_list.append(context_prompt)
        
        return context_prompt_list


def construct_context_prompt(input, topk_list, num_prompt_examples=0, remove_duplicate_ctx=False, tokenizer=None):

    query = input['question']
    prompt_question = 'Q: ' + query + '\n'


    prompt_ctxs = ''
    cnt = 0
    duplicate_flag = 0
    if remove_duplicate_ctx:
        golden_ctx = input['ctxs']['text']
        for each in topk_list[::-1]:
            if each['ctxs']['text'] == golden_ctx:
                print("REMOVE DUPLICATE!")
                duplicate_flag = 1 
                continue
            else:
                if cnt < num_prompt_examples:
                    each_prompt_question = 'Q: ' + each['question'] + '\n'
                    each_prompt_ctx = 'A: ' + each['ctxs']['title'] + ' ' + each['ctxs']['text'] + '\n\n'
                    prompt_ctxs = each_prompt_question + each_prompt_ctx + prompt_ctxs
                    cnt += 1
                else:
                    break
    else:
        for each in topk_list[:num_prompt_examples]:
            each_prompt_question = 'Q: ' + each['question'] + '\n'
            each_prompt_ctx = 'A: ' + each['ctxs']['title'] + ' ' + each['ctxs']['text'] + '\n\n'
            prompt_ctxs += each_prompt_question + each_prompt_ctx
    
    prompt_ctxs += prompt_question

    # check the length of prompt_ctxs, if it is longer than 2048, then we need to truncate from the left.

    if tokenizer is not None:
        actual_len = check_context_length(prompt_ctxs, tokenizer)
        if actual_len <= 2048:
            print("the length is {} smaller than 2048 ".format(actual_len))
            return prompt_ctxs
        else:
            print("truncate the context since it's length {} is longer than 2048 ".format(actual_len))
            prompt_ctxs = ''
            for each in topk_list[1:num_prompt_examples]:
                each_prompt_question = 'Q: ' + each['question'] + '\n'
                each_prompt_ctx = 'A: ' + each['ctxs']['title'] + ' ' + each['ctxs']['text'] + '\n\n'
                prompt_ctxs += each_prompt_question + each_prompt_ctx

            prompt_ctxs += prompt_question
            return prompt_ctxs


def check_context_length(prompt, tokenizer):
    '''check the length of prompt after tokenization, if it is too long, than 2048, we need to truncate from the left'''

    prompt_id = tokenizer.tokenize(prompt)

    return len(prompt_id)



def construct_input_prompt_for_api_call(input_list, prompt_data, retriever=None):
    """construct the prompt for context prompting-based generation"""

    args = get_args()

    # step1: sample selection
    list_of_prompt_sample_list = []
    list_of_scores = []
    
    # retrieve the k-samples    
    for input in input_list:
        if args.query_type == 'question':
            query = input['question']
        elif args.query_type == 'context':
            query = input['ctxs']['title'] + ' ' + input['ctxs']['text']
        elif args.query_type == 'question_context':
            query = input['question'] + input['ctxs']['title'] + ' ' + input['ctxs']['text']

        scores = []
        if args.with_context:
            if args.use_golden:
                prompt_sample_list, scores = prompt_sample_selection(prompt_data, query, \
                    args.num_prompt_examples, args.is_random, retriever)
            else:
                prompt_sample_list, scores = prompt_sample_selection(prompt_data, query, \
                    args.num_prompt_examples + args.shift_steps + 1, args.is_random, retriever)
        else:
            prompt_sample_list = prompt_sample_selection(prompt_data, query, \
                args.num_prompt_examples, is_random=True)

        list_of_prompt_sample_list.append(prompt_sample_list)
        list_of_scores.append(scores)

    # prepare the context
    if args.is_context_generated:
        context_prompt_list = context_construction_for_api_call(input_list, list_of_prompt_sample_list, args.num_prompt_examples)
        return context_prompt_list


def step1_prepare_context():
    """Prompt a pretrained language model to generate answer"""
    
    # get tokenizer
    args = get_args()
    
    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage():
        # load the data from input and prompt file
        # raw_data = load_data(args.input_file, args.with_context)
        # prompt_data = load_data(args.prompt_file, args.with_context)
        raw_data = load_data_kilt(args.input_file, args.with_context)
        prompt_data = load_data_kilt(args.prompt_file, args.with_context)

        input_count = len(raw_data)

        print("> loading tokenizer and encoder")

        query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                        'facebook/dpr-question_encoder-multiset-base')
        query_encoder = DPRQuestionEncoder.from_pretrained(
                "facebook/dpr-question_encoder-multiset-base").cuda()
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                            "facebook/dpr-ctx_encoder-multiset-base")
        ctx_encoder = DPRContextEncoder.from_pretrained(
                        "facebook/dpr-ctx_encoder-multiset-base").cuda()

        retriever = MyRetriever(query_encoder,
            query_tokenizer,
            ctx_encoder,
            ctx_tokenizer,
            data_list = prompt_data,
            encoded_ctx_files=args.encoded_ctx_files,
            ctx_embeddings=None,
        )

    input_pos = 0
    bz = args.micro_batch_size
    start_time = time.time()
    
    context_prompt_list=[]
    # perform prompting
    while True:
        start_time = time.time()
        print("input_pos is {} and input_count is {}, and rank is {}".format(input_pos, \
            input_count, torch.distributed.get_rank()))      

        if mpu.is_pipeline_first_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0:
            start_pos = input_pos
            end_pos = input_pos + bz if input_pos + bz < input_count else input_count
            input_list = raw_data[start_pos: end_pos]
            context_prompt_batch = \
                    construct_input_prompt_for_api_call(input_list, prompt_data, \
                                                retriever = retriever,
                                                )
            # convert the list into dictionary
            for i, each in enumerate(context_prompt_batch):
                context_prompt_dict = {}
                context_prompt_dict[start_pos + i] = each
                context_prompt_list.append(context_prompt_dict)

            input_pos += len(context_prompt_batch)


            if input_pos % 100 == 0:
                print_rank_0("input_pos: {}".format(input_pos))
            
        if input_pos == input_count:
            print("Rank {} finished the context genration in {} seconds !".format(torch.distributed.get_rank(), \
                time.time()- start_time))
            break    

    print("write the context prompt to file {}".format(args.save_context_prompt_path))
    with open(args.save_context_prompt_path, 'w') as fcontxt_out: 
        for context_prompt in context_prompt_list:
            json.dump(context_prompt, fcontxt_out)
            fcontxt_out.write("\n")

    return


def main():
    
    args = get_args()
    
    if args.api_prompt:
        step1_prepare_context()
    



