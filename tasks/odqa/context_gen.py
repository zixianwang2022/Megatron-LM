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
from .data import load_data, load_data_distributed, load_piQA_data
from .retriever import MyRetriever
from .utils import write_output
import random
import os.path
from pathlib import Path
import shutil
import time
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder




def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


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
        return retriever.get_topk(query, k, args.emb_type)
        # return retriever.get_topk(query, k)

def post_process_generations(generations, min_token_length=5, sep='\n'):
    # return the first string that has length longer than 5
    generations_split = generations.split(sep)
    for each in generations_split:
        if len(each.strip()) >= min_token_length:
            return each.strip()
    
    return "No proper answer!"


def context_generation(input_list, list_of_topk_list, num_prompt_examples=0, gen_model=None,):

    args = get_args()

    # return the generated context
    context_prompt_list = []
    context_prompt_len_list=[]
    generation_list = []

    if os.path.exists(args.save_context_path):
        print("loading the context_gen_file from {}".format(args.save_context_path))
        with open(args.save_context_path, 'r') as f:
            ctx_generation_list = f.readlines()
        for input in input_list:
            generation_list.append(ctx_generation_list[int(input['id'])])
        return generation_list

    else:
        for input, topk_list in zip(input_list,list_of_topk_list):
            context_prompt = construct_context_prompt(input['question'], topk_list, num_prompt_examples)
            context_prompt_len_list.append(len(context_prompt))
            context_prompt_list.append(context_prompt)
        
        input_pos = 0
        input_count = len(input_list)

        assert args.save_context_path is not None, 'the save_context_path should not be None'

        batch_size = args.micro_batch_size
        
        while True:
                start_pos = input_pos
                end_pos = input_pos + batch_size if input_pos + batch_size < input_count else input_count
                context_prompt_batch = context_prompt_list[start_pos: end_pos]

                input_pos += end_pos - start_pos

                outputs_batch = generate_and_post_process(
                            model=gen_model, 
                            prompts=context_prompt_batch, 
                            tokens_to_generate=100,
                            top_k_sampling=0,
                            top_p_sampling=0.9,
                            temperature = args.temperature)

                # try beam_search
                # assert len(context_prompt_batch) == 1 
                # outputs_batch = beam_search_and_post_process(
                #             model=gen_model, 
                #             prompts=context_prompt_batch, 
                #             tokens_to_generate = 100,
                #             beam_size=4,
                #             )


                prompts_plus_generations_list = outputs_batch[0]

                # write the generated output to the output file
                if mpu.get_tensor_model_parallel_rank() == 0:
                    if mpu.is_pipeline_first_stage():
                        for prompts_plus_generations, raw_text_len in zip(prompts_plus_generations_list, context_prompt_len_list):
                            generations = prompts_plus_generations[raw_text_len:].strip()
                            generations_str = post_process_generations(generations, min_token_length=5, sep='\n')
                            generation_list.append(generations_str)
                
                if input_pos == input_count:
                    # print("Rank {} finished the genration!".format(torch.distributed.get_rank()), flush=True)
                    return generation_list
 

def construct_context_prompt(query, topk_list, num_prompt_examples=0):

    prompt_question = 'Q: ' + query + '\n'

    prompt_ctxs = ''
    for each in topk_list[:num_prompt_examples]:
        each_prompt_question = 'Q: ' + each['question'] + '\n'
        each_prompt_ctx = 'A: ' + each['ctxs']['title'] + ' ' + each['ctxs']['text'] + '\n\n'
        prompt_ctxs += each_prompt_question + each_prompt_ctx
    
    prompt_ctxs += prompt_question

    return prompt_ctxs
 

def construct_input_prompt_ours(input_list, prompt_data, retriever=None, model=None):
    """construct the prompt for context prompting-based generation"""

    args = get_args()

    # step1: sample selection
    list_of_prompt_sample_list = []

    

    for input in input_list:
        if args.query_type == 'question':
            query = input['question']
        elif args.query_type == 'context':
            query = input['ctxs']['title'] + ' ' + input['ctxs']['text']
        elif args.query_type == 'question_context':
            query = input['question'] + input['ctxs']['title'] + ' ' + input['ctxs']['text']

        if args.with_context:
            if args.use_golden:
                prompt_sample_list= prompt_sample_selection(prompt_data, query, \
                    args.num_prompt_examples, args.is_random, retriever)
            else:
                prompt_sample_list= prompt_sample_selection(prompt_data, query, \
                    args.num_prompt_examples + args.shift_steps + 1, args.is_random, retriever)
        else:
            prompt_sample_list= prompt_sample_selection(prompt_data, query, \
                args.num_prompt_examples, is_random=True)
 
        list_of_prompt_sample_list.append(prompt_sample_list)

    # step2: context generation
    context_current_list = []
    # generate the context
    if args.is_context_generated:
        print("Using the generated passage as context!")
        assert model is not None, 'The model used for context generation should not be None!'
        context_current_list = context_generation(input_list, list_of_prompt_sample_list, args.num_prompt_examples, model)
    else: 
        print("Using the retrieved/golden passage as context!")
        for input, prompt_sample_list in zip(input_list, list_of_prompt_sample_list):
            context_current=""
            if args.use_golden:
                context_current = input['ctxs']['title'] + ' ' + input['ctxs']['text']
            else:
                context_current = prompt_sample_list[0]['ctxs']['title'] + ' ' + prompt_sample_list[0]['ctxs']['text']
            
            context_current_list.append(context_current)


    prompt_text_list = []
    raw_text_len_list = []

    for input, prompt_sample_list, context_current in zip(input_list, list_of_prompt_sample_list, context_current_list):
        # prepare the prompt_question
        prompt_text, prompt_question = '', ''
        if args.with_context:

            if args.num_prompt_examples == 0:
                prompt_question = 'Context: ' + context_current + '\n' + 'Question: ' + input['question'] + '\n' + 'Answer:'  
            else:
                prompt_question = 'Context: ' + context_current + '\n' + 'Question: ' + input['question'] + '\n'

            # prepare the prompt_text
            if not args.use_golden and args.shift_steps:
                prompt_sample_list = prompt_sample_list[args.shift_steps:]

            for each in prompt_sample_list[:args.num_prompt_examples]:
                answer=''
                prompt_text_tmp = ''
                if 'target' in each:
                    answer = each['target']
                else:
                    answer = each['answers'][0]
                context_each = each['ctxs']['title'] + ' ' + each['ctxs']['text']
                prompt_text_tmp = 'Context: ' + context_each + '\n' + 'Question: ' + each['question'] + '\n' + 'Answer: ' + answer + '\n'
                
                prompt_text += prompt_text_tmp

        else:

            if args.num_prompt_examples == 0:
                prompt_question = 'Question: ' + input['question'] + '\n' + 'Answer:'  
            else:
                prompt_question = 'Question: ' + input['question'] + '\n'

            for each in prompt_sample_list:
                answer=''
                if 'target' in each:
                    answer = each['target']
                else:
                    answer = each['answers'][0]
                
                prompt_text += 'Question: ' + each['question'] + '\n' + 'Answer: ' + answer + '\n'

        prompt_text += prompt_question
        prompt_text_list.append(prompt_text)
        raw_text_len = len(prompt_text)
        raw_text_len_list.append(raw_text_len)

    
    return prompt_text_list, raw_text_len_list, context_current_list


def batch_generate_samples_by_prompting_input_from_file_new(model):
    """Prompt a pretrained language model to generate answer"""
    
    # get tokenizer
    args = get_args()
    
    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage():
        # load the data from input and prompt file
        raw_data = load_data(args.input_file, args.with_context)
        prompt_data = load_data(args.prompt_file, args.with_context)
        input_count = len(raw_data)

        if args.output_file is None:
            output_file = args.input_file + ".out"
            print('`output-file` not specified, setting '
                    'it to {}'.format(output_file))
        else:
            output_file = args.output_file
            print("output_file is {}".format(output_file))

        print("> loading tokenizer and encoder")
        # query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        #                 'facebook/dpr-question_encoder-single-nq-base')
        # query_encoder = DPRQuestionEncoder.from_pretrained(
        #         "facebook/dpr-question_encoder-single-nq-base").cuda()
        # ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        #                     "facebook/dpr-ctx_encoder-single-nq-base")
        # ctx_encoder = DPRContextEncoder.from_pretrained(
        #                 "facebook/dpr-ctx_encoder-single-nq-base").cuda()

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
    model.eval()
    start_time = time.time()
    cnt = 0
    
    context_list = []

    # perform prompting
    with torch.no_grad():
        with open(output_file, "w") as fname_out:
            while True:
                start_time = time.time()
                print("input_pos is {} and input_count is {}, and rank is {}".format(input_pos, \
                    input_count, torch.distributed.get_rank()), flush=True)      
                
                if mpu.is_pipeline_first_stage() \
                and mpu.get_tensor_model_parallel_rank() == 0:
                    start_pos = input_pos
                    end_pos = input_pos + bz if input_pos + bz < input_count else input_count
                    input_list = raw_data[start_pos: end_pos]
                    prompt_text_list, raw_text_len_list, context_current_list = \
                        construct_input_prompt_ours(input_list, prompt_data, \
                                                        retriever = retriever,\
                                                        model=model,
                                                        )

                    context_list.extend(context_current_list)

                    if input_pos < int(args.micro_batch_size) * 5:
                        print("======samples=====!")
                        print(prompt_text_list[0])                                    
                    

                    input_pos += len(prompt_text_list)
                    
                    if input_pos % 100 == 0:
                        print_rank_0("input_pos: {}".format(input_pos))


                if args.openai_api:
                    assert args.engine is not None
                    print("input is '{}'".format(prompt_text_list[0]))
                    api_text_list = [item.strip() for item in prompt_text_list]
                    results = call_openai_api(api_text_list, engine=args.engine)
                    for item in results:
                        cnt += 1
                        generations_str = item['text']
                        print("output is ", item['text'])
                        fname_out.write(generations_str)
                        fname_out.write("\n")
                        if cnt % 100 == 0:
                            print("{} examples need {}".format(cnt, time.time() - start_time))
                else:
                    # outputs = generate_and_post_process(
                    #             model=model, 
                    #             prompts=prompt_text_list, 
                    #             tokens_to_generate=args.out_seq_length,
                    #             top_k_sampling=args.top_k_sampling,
                    #             top_p_sampling=args.top_p_sampling,
                    #             temperature = args.temperature)

                    # try beam_search
                    outputs = beam_search_and_post_process(
                                model=model, 
                                prompts=prompt_text_list, 
                                tokens_to_generate = args.out_seq_length,
                                beam_size=4,
                                )


                    prompts_plus_generations_list = outputs[0]

                    # write the generated output to the output file
                    if mpu.get_tensor_model_parallel_rank() == 0:
                        if mpu.is_pipeline_first_stage():                            
                            for prompts_plus_generations, raw_text_len in zip(prompts_plus_generations_list, raw_text_len_list):
                                generations = prompts_plus_generations[raw_text_len:].strip()
                                generations_str = post_process_generations(generations, min_token_length=5, sep='\n')
                                fname_out.write(generations_str)
                                fname_out.write("\n")
                
                if input_pos == input_count:
                    print("Rank {} finished the genration in {} seconds !".format(torch.distributed.get_rank(), \
                        time.time()- start_time), flush=True)
                    break

        if os.path.exists(args.save_context_path) == False:
            with open(args.save_context_path, 'w') as fcontxt_out: 
                if mpu.get_tensor_model_parallel_rank() == 0 \
                                and mpu.is_pipeline_first_stage():
                                    for context_generation in context_list:
                                        fcontxt_out.write(context_generation)
                                        fcontxt_out.write('\n')
        
        return


def batch_generate_context(model):
    """Prompt a pretrained language model to generate answer"""
    
    # get tokenizer
    args = get_args()
    
    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage():
        # load the data from input and prompt file
        raw_data = load_data(args.input_file, args.with_context)
        prompt_data = load_data(args.prompt_file, args.with_context)
        input_count = len(raw_data)

        if args.output_file is None:
            output_file = args.input_file + ".out"
            print('`output-file` not specified, setting '
                    'it to {}'.format(output_file))
        else:
            output_file = args.output_file
            print("output_file is {}".format(output_file))

        print("> loading tokenizer and encoder")
        # query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        #                 'facebook/dpr-question_encoder-single-nq-base')
        # query_encoder = DPRQuestionEncoder.from_pretrained(
        #         "facebook/dpr-question_encoder-single-nq-base").cuda()
        # ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        #                     "facebook/dpr-ctx_encoder-single-nq-base")
        # ctx_encoder = DPRContextEncoder.from_pretrained(
        #                 "facebook/dpr-ctx_encoder-single-nq-base").cuda()

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
    model.eval()
    start_time = time.time()
    cnt = 0
    
    context_list = []
    # perform prompting
    with torch.no_grad():
        with open(output_file, "w") as fname_out:
                while True:
                    start_time = time.time()
                    print("input_pos is {} and input_count is {}, and rank is {}".format(input_pos, \
                        input_count, torch.distributed.get_rank()), flush=True)      
                    
                    if mpu.is_pipeline_first_stage() \
                    and mpu.get_tensor_model_parallel_rank() == 0:
                        start_pos = input_pos
                        end_pos = input_pos + bz if input_pos + bz < input_count else input_count
                        input_list = raw_data[start_pos: end_pos]
                        prompt_text_list, _, context_current_list = construct_input_prompt_ours(input_list, prompt_data, \
                                                            retriever = retriever,\
                                                            model=model,
                                                            )
                        context_list.extend(context_current_list)

                        if input_pos < int(args.micro_batch_size) * 5:
                            print("======samples=====!")
                            print(prompt_text_list[0])                                    
                        
                        input_pos += len(prompt_text_list)
                        
                        if input_pos % 100 == 0:
                            print_rank_0("input_pos: {}".format(input_pos))
                    
                    if input_pos == input_count:
                        print("Rank {} finished the context genration in {} seconds !".format(torch.distributed.get_rank(), \
                            time.time()- start_time), flush=True)
                        break    

        if os.path.exists(args.save_context_path) == False:
            print("write the generated context to file {}".format(args.save_context_path))
            with open(args.save_context_path, 'w') as fcontxt_out: 
                if mpu.get_tensor_model_parallel_rank() == 0 \
                                and mpu.is_pipeline_first_stage():
                                    for context_generation in context_list:
                                        fcontxt_out.write(context_generation)
                                        fcontxt_out.write('\n')
        
        return



def main():

    args = get_args()
    
    random.seed(1234)

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # Set up model and load checkpoint.
    model = get_model(model_provider, wrap_with_ddp=False)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # perform the prompting
    batch_generate_samples_by_prompting_input_from_file_new(model)
    # batch_generate_context(model)

    # for PIQA, need to merge with other functions later
    # batch_generate_samples_by_prompting_input_from_file_for_piQA(model)

