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
from .data import load_data, load_data_distributed, load_piQA_data, load_data_qg, load_data_dpr_wq,load_data_kilt
from .retriever import MyRetriever
from .utils import write_output, truncate_input, check_context_length
import random
import os.path
from pathlib import Path
import shutil
import time
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder

from .question_gen import generate_question



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


def call_model_api(inputs, tokens_to_generate, top_k_sampling,\
                                    top_p_sampling,temperature, random_seed):
    """Calling the model api to get the output generations"""
    
    args = get_args()

    # The following is an example of using the Megatron API
    # You can also implement your own API function to place this part
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {"prompts": [inputs], \
            "tokens_to_generate": tokens_to_generate, \
            "top_k_sampling": 0, \
            "top_p_sampling": top_p_sampling, \
            "temperature": temperature, \
            "random_seed": random_seed, \
            }
    data_json = json.dumps(data)
    outputs = requests.put(args.megatron_api_url, headers=headers, data=data_json).json()["text"][0]

    input_len = len(inputs)
    outputs = outputs[input_len:]
    outputs = outputs.split("\n")[0].strip()
    
    return outputs


def generate_samples_by_calling_api(prompts, tokens_to_generate, top_k_sampling,\
                                    top_p_sampling,temperature, random_seed):
    """ Generate outputs by calling"""
    args = get_args()
    # call the api to get the output generations
    generations = call_model_api(prompts, tokens_to_generate, top_k_sampling,\
                                    top_p_sampling,temperature, random_seed)
    
    return generations

def prompt_sample_selection(data_list, query = "", k=10, is_random=True, retriever=None):

    args = get_args()

    if k==0:
        return []

    if is_random:
        print("random select the top-k samples")
        return random.sample(data_list, k), []
    else: 
        ## option1: return the top-k
        assert retriever is not None
        # print("select the samples based on query : {} similarity!".format(args.emb_type))
        if args.remove_duplicate_ctx:
            return retriever.get_topk(query, k+10, args.emb_type)
        else:
            return retriever.get_topk(query, k, args.emb_type)

def post_process_generations(generations, min_token_length=5, sep='\n'):
    # return the first string that has length longer than 5
    generations_split = generations.split(sep)
    for each in generations_split:
        if len(each.strip()) >= min_token_length:
            return each.strip()
    
    return "No proper answer!"



def post_process_generations_with_positions(generations, min_token_length=5, sep='\n'):
    # return the first string that has length longer than 5
    generations = generations.replace("A: ", "")
    generations = generations.replace("Answer: ", "")

    generations_split = generations.split(sep)
    start_pos, end_pos = 0,0
    for each in generations_split:
        # if len(each.strip()) >= min_token_length + len('Answer: '):
        if len(each.strip()) >= min_token_length:
            end_pos += len(each)
            return each.strip(), (start_pos, end_pos)
        else:
            start_pos += len(each) + len(sep)
            end_pos += len(each) + len(sep)

    
    return "No proper answer!", (0, 0)

def context_read(input_list, ctx_generation_list):
    generation_list = []
    for input in input_list:
        print("====")
        print(int(input['id']))
        generation_list.append(ctx_generation_list[int(input['id'])].strip())
    return generation_list


def context_generation(input_list, list_of_topk_list, num_prompt_examples=0, gen_model=None, multiple_gen=1):

    args = get_args()

    # return the generated context
    context_prompt_list = []
    context_prompt_len_list=[]
    generation_list = []

    for input, topk_list in zip(input_list,list_of_topk_list):
        context_prompt = construct_context_prompt(input, topk_list, num_prompt_examples, args.remove_duplicate_ctx)
        context_prompt_len_list.append(len(context_prompt))
        context_prompt_list.append(context_prompt)
    
    input_pos = 0
    input_count = len(input_list)

    assert args.save_context_path is not None, 'the save_context_path should not be None'

    batch_size = args.micro_batch_size
    
    while True:
        if mpu.is_pipeline_first_stage():
            start_pos = input_pos
            end_pos = input_pos + batch_size if input_pos + batch_size < input_count else input_count
            context_prompt_batch = context_prompt_list[start_pos: end_pos]
            input_pos += end_pos - start_pos
            prompts_plus_generations_list =[]
            
        for i in range(multiple_gen):
            print("======")
            print("generating the {}-th round".format(i))

            outputs_batch = generate_and_post_process(
                        model=gen_model, 
                        prompts=context_prompt_batch, 
                        tokens_to_generate=100,
                        top_k_sampling=0,
                        top_p_sampling=0.9,
                        temperature = args.temperature,
                        random_seed=args.random_seed + i*10
                        )

            prompts_plus_generations_list = outputs_batch[0]

                # write the generated output to the output file
            # if mpu.get_tensor_model_parallel_rank() == 0 \
            if mpu.is_pipeline_first_stage():
                for prompts_plus_generations, raw_text_len in zip(prompts_plus_generations_list, context_prompt_len_list):
                    generations = prompts_plus_generations[raw_text_len:].strip()
                    generations_str = post_process_generations(generations, min_token_length=5, sep='\n')
                    generation_list.append(generations_str)
                
        if input_pos == input_count:
            # print("Rank {} finished the genration!".format(torch.distributed.get_rank()), flush=True)
            return generation_list
 

def construct_context_prompt(input, topk_list, num_prompt_examples=0, remove_duplicate_ctx=False):

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

    return prompt_ctxs
 

def construct_input_prompt_ours(input_list, prompt_data, ctx_generation_list=None, retriever=None, model=None, \
                                tokenizer=None ,multiple_gen=1):
    """construct the prompt for context prompting-based generation"""

    args = get_args()

    # step1: sample selection
    list_of_prompt_sample_list = []
    list_of_scores = []
    
    # prepare the k-samples by generation
    if args.question_generation:
        #randomly select 10 examples, and fix them for all question generations

        for input in input_list:
            prompt_sample_list = prompt_sample_selection(prompt_data, query='', \
                    k=args.num_prompt_examples, is_random=True)

            question_context_pairs,scores = generate_question(input, gen_model=model,prompt_list=prompt_sample_list, num_prompt_examples=10)
            list_of_prompt_sample_list.append(question_context_pairs)
            list_of_scores.append(scores)
    # retrieve the k-samples    
    else:
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
                    # prompt_sample_list, scores = prompt_sample_selection(prompt_data, query, \
                    #     args.num_prompt_examples + args.shift_steps + 1, args.is_random, retriever)
                    prompt_sample_list, scores = prompt_sample_selection(prompt_data, query, \
                        args.num_prompt_examples + args.shift_steps, args.is_random, retriever)

            else:
                prompt_sample_list = prompt_sample_selection(prompt_data, query, \
                    args.num_prompt_examples, is_random=True)
    
            list_of_prompt_sample_list.append(prompt_sample_list)
            list_of_scores.append(scores)

    # step2: context generation
    context_current_list = []
    # generate the context
    if args.is_context_generated:
        print("Using the generated passage as context!")
        if args.api_prompt == False:
            assert model is not None, 'The model used for context generation should not be None!'
            model.eval()

        if ctx_generation_list is not None:
            context_current_list = context_read(input_list, ctx_generation_list)
        else:
            context_current_list = context_generation(input_list, list_of_prompt_sample_list, args.num_prompt_examples, model, multiple_gen)
    else: 
        print("Using the retrieved/golden passage as context!")
        if args.use_wiki_samples:
            print("using the wikipedia retrieved top-1 as C_gen!")
            for input in input_list:
                top1 = input['output'][0]['provenance'][0]
                context_current = top1['wikipedia_title'] + ' ' + top1['text']
                context_current_list.append(context_current)
        else:
            for input, prompt_sample_list in zip(input_list, list_of_prompt_sample_list):
                context_current=""
                if args.use_golden:
                    print("using the golden context as C_gen!")
                    context_current = input['ctxs']['title'] + ' ' + input['ctxs']['text']
                else:
                    # print("using the top-1 context as C_gen!")
                    # # context_current = prompt_sample_list[0]['ctxs']['title'] + ' ' + prompt_sample_list[0]['ctxs']['text']
                    # context_current = prompt_sample_list[-1]['ctxs']['title'] + ' ' + prompt_sample_list[-1]['ctxs']['text']

                    print("using the top-{} context as C_gen!".format(args.kth_context_from_retrieval))
                    context_current = prompt_sample_list[-args.kth_context_from_retrieval]['ctxs']['title'] + ' ' + prompt_sample_list[-args.kth_context_from_retrieval]['ctxs']['text']

                context_current_list.append(context_current)


    prompt_text_list = []
    raw_text_len_list = []

    # print_rank_0("=="*10)
    # print_rank_0(context_current_list[0])
    if args.question_generation:
        return [], [], context_current_list, list_of_prompt_sample_list, list_of_scores

    # if mpu.get_tensor_model_parallel_rank() == 0 \
    if mpu.is_pipeline_first_stage:
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

            # this is to guarantee the length is within 2048, since the megatron model do not check will rasie an error.
            if tokenizer is not None:
                prompt_text = truncate_input(prompt_text, tokenizer)

            prompt_text_list.append(prompt_text)
            raw_text_len = len(prompt_text)
            raw_text_len_list.append(raw_text_len)
    
    # print(prompt_text_list[0],flush=True)
    return prompt_text_list, raw_text_len_list, context_current_list, list_of_prompt_sample_list, list_of_scores


def batch_generate_samples_by_prompting_input_from_file_new(model):
    """Prompt a pretrained language model to generate answer"""
    
    # get tokenizer
    args = get_args()

    megatron_tokenizer =  get_tokenizer()
    megatron_tokenizer.pad_token = "<|endoftext|>"

    assert megatron_tokenizer.tokenize('hello\n\nhello') == [31373, 198, 198, 31373]

    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage():
        if 'WebQuestions' in args.input_file:
            raw_data = load_data_kilt(args.input_file, answer_filtering=False)
            prompt_data = load_data_kilt(args.prompt_file, answer_filtering=False)
        else:
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
            # ctx_embeddings=None,
        )

        input_pos = 0

        bz = args.micro_batch_size
        model.eval()
        start_time = time.time()
        cnt = 0
        
        context_list = []
        topk_list=[]
        scores_list =[]

        ctx_generation_list=None

        if args.save_context_path is not None:
            if '530' in args.save_context_path:
                if os.path.exists(args.save_context_path) and os.path.getsize(args.save_context_path) > 0:
                    import csv
                    with open(args.save_context_path, "r") as fin:
                        wr = csv.reader(fin)
                        context_data = list(wr)
                    ctx_generation_list = []
                    for each in context_data:
                        ctx_generation_list.append(each[1])
                    print("Directly read the currect context data from {}, and the sample is:".format(args.save_context_path))
                    print(ctx_generation_list[0])
            else:
                if os.path.exists(args.save_context_path):
                    print("loading the context_gen_file from {}".format(args.save_context_path))
                    with open(args.save_context_path, 'r') as f:
                        ctx_generation_list = f.readlines()

    # perform prompting
    with torch.no_grad():
        # if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        #     print("File {} already exists!".format(output_file))
        #     fname_out = open(output_file, "a")
        #     current_data = fname_out.readlines()
        #     input_pos = len(current_data)
        #     print("Started from the {} example".format(input_pos))
        # else:
        #     fname_out = open(output_file, "w")
        fname_out = open(output_file, "w")
        while True:
            print_rank_0("input_pos is {} and input_count is {}, and rank is {}".format(input_pos, \
                input_count, torch.distributed.get_rank()))      
            
            if mpu.is_pipeline_first_stage():
            # and mpu.get_tensor_model_parallel_rank() == 0:
                start_pos = input_pos
                if input_pos + bz < input_count:
                    end_pos = input_pos + bz 
                else:
                    end_pos = input_count
                input_list = raw_data[start_pos: end_pos]

                prompt_text_list, raw_text_len_list, context_current_list, \
                    list_of_prompt_sample_list, list_of_scores = \
                    construct_input_prompt_ours(input_list, prompt_data, \
                                                ctx_generation_list, \
                                                retriever = retriever,\
                                                model=model,
                                                tokenizer=megatron_tokenizer,
                                                )

                context_list.extend(context_current_list)
                topk_list.extend(list_of_prompt_sample_list)
                scores_list.extend(list_of_scores)

                if input_pos < int(args.micro_batch_size) * 5:
                    print("======samples=====!", flush=True)
                    print(prompt_text_list[0], flush=True) 
                    print("rank is {}".format(torch.distributed.get_rank()), flush=True)                                
                
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
                if args.beam_search:
                    outputs = beam_search_and_post_process(
                                model=model, 
                                prompts=prompt_text_list, 
                                tokens_to_generate=args.out_seq_length,
                                beam_size=args.beam_size,
                                )
                    prompts_plus_generations_list = outputs[0]
                    prompts_plus_generations_prob_list = outputs[2]  # scores, though we wont use it
                else:
                    outputs = generate_and_post_process(
                                model=model, 
                                prompts=prompt_text_list, 
                                tokens_to_generate=args.out_seq_length,
                                top_k_sampling=args.top_k_sampling,
                                top_p_sampling=args.top_p_sampling,
                                temperature = args.temperature,
                                return_output_log_probs=True,
                                )


                    prompts_plus_generations_list = outputs[0]
                    prompts_plus_generations_prob_list = outputs[2]
                    # prompts_plus_generations_token_list = outputs[3]

                # write the generated output to the output file
                if mpu.get_tensor_model_parallel_rank() == 0:
                    if mpu.is_pipeline_first_stage():                            
                        for prompts_plus_generations, raw_text_len, logprob in zip(prompts_plus_generations_list, \
                                                raw_text_len_list, prompts_plus_generations_prob_list):
                            generations = prompts_plus_generations[raw_text_len:].strip()
                            
                            if args.with_answer_probability and not args.beam_search:
                                generations_str, (start_pos, end_pos) = post_process_generations_with_positions(generations, min_token_length=5, sep='\n')
                                # print("===="*10)
                                context_enc = megatron_tokenizer.tokenize(prompts_plus_generations[:raw_text_len + start_pos])
                                context_answer_enc = megatron_tokenizer.tokenize(prompts_plus_generations[:raw_text_len + end_pos])

                                context_len = len(context_enc)
                                context_answer_len = len(context_answer_enc)
                                avg_log_prob1 = sum(logprob[context_len: context_answer_len])

                                fname_out.write(generations_str + '\t' + str(avg_log_prob1))
                                fname_out.write("\n")
                                fname_out.flush()
                            else:
                                generations_str = post_process_generations(generations, min_token_length=5, sep='\n')
                                fname_out.write(generations_str)
                                fname_out.write("\n")
                                fname_out.flush()
            
            if input_pos == input_count:
                print("Rank {} finished the genration in {} seconds !".format(torch.distributed.get_rank(), \
                    time.time()- start_time), flush=True)
                break

        if args.save_context_path is not None and os.path.exists(args.save_context_path) == False:
            print("write the generated context to file {}".format(args.save_context_path))
            with open(args.save_context_path, 'w') as fcontxt_out: 
                if mpu.get_tensor_model_parallel_rank() == 0 \
                                and mpu.is_pipeline_first_stage():
                                    for context_generation in context_list:
                                        fcontxt_out.write(context_generation)
                                        fcontxt_out.write('\n')

        if args.save_topk_context_path is not None and os.path.exists(args.save_topk_context_path) == False:
            if mpu.get_tensor_model_parallel_rank() == 0 \
                                and mpu.is_pipeline_first_stage():
                data_list = save_topk_context(topk_list, scores_list, context_list, raw_data)
                with open(args.save_topk_context_path, 'w') as f:
                    json.dump(data_list, f, indent=4)
                
                print('write to {} finished!'.format(args.save_topk_context_path))

        return

def save_topk_context(topk_list, scores_list, context_list, raw_data_list):
    data_list=[]
    data_item={}
    for each_topk, each_scores, each_gen_context, raw_data_item in zip(topk_list, scores_list, context_list, raw_data_list):
    # C1-Ck, Q1-Qk, scores, C_gen, Q
        data_item={}
        data_item['question'] = raw_data_item['question']
        data_item['gen_ctx'] = each_gen_context
        data_item['topk'] = []
        for i, (each_topk_item, each_scores_item) in enumerate(zip(each_topk, each_scores)):
            each={}
            each['id'] = i
            each['ctx'] = each_topk_item['ctxs']
            each['question'] = each_topk_item['question']
            each['score'] = each_scores_item
            data_item['topk'].append(each)
        
        data_list.append(data_item)
    
    return data_list

def batch_generate_context(model):
    """Prompt a pretrained language model to generate answer"""
    args = get_args()
    megatron_tokenizer =  get_tokenizer()
    megatron_tokenizer.pad_token = "<|endoftext|>"
    assert megatron_tokenizer.tokenize('hello\n\nhello') == [31373, 198, 198, 31373]

    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage():
        # load the data from input and prompt file
        if args.question_generation:
            raw_data = load_data_qg(args.input_file)
            prompt_data = load_data(args.prompt_file, args.with_context)
        else:
            if 'WebQuestions' in args.input_file:
                raw_data = load_data_kilt(args.input_file, answer_filtering=False)
                prompt_data = load_data_kilt(args.prompt_file, answer_filtering=False)
            else:
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
    topk_list=[]
    scores_list =[]
    ctx_generation_list=None

    multiple_gen=1
    # perform prompting
    with torch.no_grad():
        if os.path.exists(args.save_context_path) and os.path.getsize(args.save_context_path) > 0:
            print("File {} already exists!".format(args.save_context_path))
            with open(args.save_context_path, "r") as fcontxt_out:
                current_data = fcontxt_out.readlines()
            input_pos = len(current_data)
            print("Started from the {} example".format(input_pos))
            fcontxt_out = open(args.save_context_path, "a")
        else:
            fcontxt_out = open(args.save_context_path, "w")

        # fcontxt_out = open(args.save_context_path, 'w')
        while True:
            print("input_pos is {} and input_count is {}, and rank is {}".format(input_pos, \
                input_count, torch.distributed.get_rank()), flush=True)      
    
            if mpu.is_pipeline_first_stage(): #\
            # and mpu.get_tensor_model_parallel_rank() == 0:
                start_pos = input_pos
                end_pos = input_pos + bz if input_pos + bz < input_count else input_count
                input_list = raw_data[start_pos: end_pos]
                _, _, context_current_list,\
                    list_of_prompt_sample_list, list_of_scores = \
                        construct_input_prompt_ours(input_list, prompt_data, \
                                                    ctx_generation_list, \
                                                    retriever = retriever,\
                                                    model=model, \
                                                    multiple_gen=multiple_gen, \
                                                    tokenizer=megatron_tokenizer,
                                                    )
                
                ordered_context_current_list=[]
                if multiple_gen>1:
                    for i in range(end_pos-start_pos):
                        for j in range(multiple_gen):
                            ordered_context_current_list.append(context_current_list[i+j*(end_pos-start_pos)])
                    # context_list.extend(ordered_context_current_list)
                else:
                    ordered_context_current_list=context_current_list
                    # context_list.extend(context_current_list)

                topk_list.extend(list_of_prompt_sample_list)
                scores_list.extend(list_of_scores)
                
                if input_pos < int(args.micro_batch_size) * 5:
                    print_rank_0("======generated context samples=====!")
                    print_rank_0(context_current_list[0])                                    

                input_pos += (end_pos - start_pos)

                if input_pos % 100 == 0:
                    print_rank_0("input_pos: {}".format(input_pos))
            
                if mpu.get_tensor_model_parallel_rank() == 0 \
                    and mpu.is_pipeline_first_stage():
                    for context_generation in ordered_context_current_list:
                        fcontxt_out.write(context_generation)
                        fcontxt_out.write('\n')
                        fcontxt_out.flush()

            if input_pos == input_count:
                print("Rank {} finished the context genration in {} seconds !".format(torch.distributed.get_rank(), \
                    time.time()- start_time), flush=True)
                break    

        # if args.save_context_path is not None and os.path.exists(args.save_context_path) == False:
        #     print("write the generated context to file {}".format(args.save_context_path))
        #     with open(args.save_context_path, 'w') as fcontxt_out: 
        #         if mpu.get_tensor_model_parallel_rank() == 0 \
        #                         and mpu.is_pipeline_first_stage():
        #                             for context_generation in context_list:
        #                                 fcontxt_out.write(context_generation)
        #                                 fcontxt_out.write('\n')
        #                                 fcontxt_out.flush()

        if args.save_topk_context_path is not None and os.path.exists(args.save_topk_context_path) == False:
            if mpu.get_tensor_model_parallel_rank() == 0 \
                                and mpu.is_pipeline_first_stage():
                data_list = save_topk_context(topk_list, scores_list, context_list, raw_data)
                with open(args.save_topk_context_path, 'w') as f:
                    json.dump(data_list, f, indent=4)
                
                print('write to {} finished!'.format(args.save_topk_context_path))

        return


def main():

    args = get_args()
    
    random.seed(1234)

    if args.api_prompt:
        model = None
    else:
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
    # batch_generate_samples_by_prompting_input_from_file_new(model)
    batch_generate_context(model)

    # for PIQA, need to merge with other functions later
    # batch_generate_samples_by_prompting_input_from_file_for_piQA(model)

