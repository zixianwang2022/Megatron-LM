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

"""CGAP: context generation and answer prediction for open-domain question answering"""

import json
import requests
import time
import random
from collections import Counter
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
# from megatron import get_tokenizer
from retriever import MyRetriever
from api_utils import load_data, normalize_answer, get_tasks_args


def call_model_api(inputs, tokens_to_generate, top_k_sampling,\
                                    top_p_sampling,temperature, random_seed, url):
    """Calling the model api to get the output generations"""
    # The following is an example of using the Megatron API
    # You can also implement your own API function to place this part
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {"prompts": inputs, \
            "tokens_to_generate": tokens_to_generate, \
            "top_k": 0, \
            "top_p": top_p_sampling, \
            "temperature": temperature, \
            "random_seed": random_seed, \
            }
    data_json = json.dumps(data)
    outputs = requests.put(url, headers=headers, data=data_json).json()['text'] 
    return outputs

def prompt_sample_selection(query = "", k=10, retriever=None):
    if k==0:
        return []
    assert retriever is not None
    return retriever.get_topk(query, k, 'query_ctx')

def post_process_generations(generations, min_token_length=5, sep='\n'):
    generations_split = generations.split(sep)
    for each in generations_split:
        if len(each.strip()) >= min_token_length:
            each = each.replace("A: ", "")
            each = each.replace("A:", "")
            return each.strip()
    return "No proper answer!"

def truncate_input(prompt_text, tokenizer, spliter="Question: "):
    actual_len = check_context_length(prompt_text, tokenizer)
    if actual_len <= 2048:
        return prompt_text
    else:
        prompt_text_list = prompt_text.split(spliter)
        prompt_text = spliter.join(prompt_text_list[2:])
        return truncate_input(prompt_text, tokenizer)

def check_context_length(prompt, tokenizer):
    '''check the length of prompt after tokenization, if it is too long, 
    than 2048, we need to truncate from the left'''
    prompt_id = tokenizer.tokenize(prompt)
    return len(prompt_id)

def context_generation(input_list, list_of_topk_list, \
                    megatron_api_url=None, tokenizer=None, \
                        random_seed=1234, ctx_len=100, args=None):

    context_prompt_len_list=[]
    prompts_plus_generations_list=[]
    gen_ctx_list = []

    context_prompt_list, context_prompt_len_list = \
                construct_context_prompt(input_list, list_of_topk_list, tokenizer)
    
    current_pos=0
    bz=args.micro_batch_size
    input_count=len(input_list)
    
    while True:
        start_pos=current_pos
        end_pos = current_pos + bz if current_pos + bz < input_count else input_count
        current_batch= context_prompt_list[start_pos: end_pos]
        outputs = call_model_api(
                            inputs=current_batch, 
                            tokens_to_generate=ctx_len,
                            top_k_sampling=0,
                            top_p_sampling=0.9,
                            temperature = 1,
                            random_seed= random_seed,
                            url=megatron_api_url,
                            )
        prompts_plus_generations_list.extend(outputs)  
        current_pos += len(current_batch)
        if current_pos == input_count:
            break                  
    for prompts_plus_generations, raw_text_len in zip(prompts_plus_generations_list, context_prompt_len_list):
        generations = prompts_plus_generations[raw_text_len:].strip()
        generations_str = post_process_generations(generations, min_token_length=5, sep='\n')
        gen_ctx_list.append(generations_str)
            
    return gen_ctx_list
 

def construct_context_prompt(input_list, list_of_topk_list, tokenizer=None):
    prompt_ctxs_list=[]
    raw_text_len_list = []

    for input, topk_list in zip(input_list, list_of_topk_list):
        query = input['question']
        prompt_question = 'Q: ' + query + '\n'

        prompt_ctxs = ''
        for each in topk_list:
            each_prompt_question = 'Q: ' + each['question'] + '\n'
            each_prompt_ctx = 'A: ' + each['ctxs']['title'] + ' ' + each['ctxs']['text'] + '\n\n'
            prompt_ctxs += each_prompt_question + each_prompt_ctx
        
        prompt_ctxs += prompt_question
        if tokenizer is not None:
            prompt_ctxs = truncate_input(prompt_ctxs, tokenizer, spliter='Q: ')
        
        prompt_ctxs_list.append(prompt_ctxs)
        raw_text_len_list.append(len(prompt_ctxs))

    return prompt_ctxs_list, raw_text_len_list
 
def construct_answer_prompt(input_list, list_of_prompt_sample_list, \
                            context_current_list, tokenizer=None):
    prompt_text_list = []
    raw_text_len_list = []
    for input, prompt_sample_list, context_current in zip(input_list, \
                                    list_of_prompt_sample_list, context_current_list):
        prompt_text, prompt_question = '', ''
        prompt_question = 'Context: ' + context_current + '\n' + 'Question: ' + input['question'] + '\n'
        for each in prompt_sample_list:
            answer=''
            prompt_text_tmp = ''
            if 'target' in each:
                answer = each['target']
            else:
                answer = each['answers'][0]
            context_each = each['ctxs']['title'] + ' ' + each['ctxs']['text']
            prompt_text_tmp = 'Context: ' + context_each + '\n' + 'Question: ' + each['question'] \
                + '\n' + 'Answer: ' + answer + '\n'
            
            prompt_text += prompt_text_tmp

        prompt_text += prompt_question
        if tokenizer is not None:
            prompt_text = truncate_input(prompt_text, tokenizer)

        prompt_text_list.append(prompt_text)
        raw_text_len = len(prompt_text)
        raw_text_len_list.append(raw_text_len)
    
    return prompt_text_list, raw_text_len_list

def answer_prediction(input_list, list_of_prompt_sample_list, context_current_list, tokenizer=None, args=None):
    print("using the megatron API to predict answer!")
    answers_list=[]
    prompt_text_list, raw_text_len_list = \
        construct_answer_prompt(input_list, list_of_prompt_sample_list, context_current_list, tokenizer)
    current_pos=0
    bz=args.micro_batch_size
    input_count=len(input_list)
    prompts_plus_generations_list=[]

    while True:
        start_pos=current_pos
        end_pos = current_pos + bz if current_pos + bz < input_count else input_count
        current_batch= prompt_text_list[start_pos: end_pos]
        outputs_batch = call_model_api(
                            inputs=current_batch, 
                            tokens_to_generate=args.out_seq_length,
                            top_k_sampling=args.top_k_sampling,
                            top_p_sampling=args.top_p_sampling,
                            temperature = args.temperature,
                            random_seed=args.random_seed,
                            url=args.megatron_api_url,
                            )
        prompts_plus_generations_list.extend(outputs_batch)
        current_pos += len(current_batch)
        if current_pos == input_count:
            break

    for prompts_plus_generations, raw_text_len in zip(prompts_plus_generations_list, raw_text_len_list):
        generations = prompts_plus_generations[raw_text_len:].strip()
        generations_str = post_process_generations(generations, min_token_length=5, sep='\n')
        answers_list.append(generations_str)

    return answers_list

def marginalize_prediction(answer_list):
    normalized_answer_list = []
    for answer in answer_list:
        answer = answer.replace("Answer:","")
        answer = answer.replace("Answer: ","")
        answer = answer.replace('????  ', "")
        answer = answer.replace('A: ',"")
        answer = answer.replace("A:", "")
        answer = answer.strip()

        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")
        answer = normalize_answer(answer) # normalize the answer
        normalized_answer_list.append(answer)
    
    x = Counter(normalized_answer_list)
    (most_predicted_answer, _) = x.most_common()[0]
    
    return most_predicted_answer

def cgap(input, margin_number, ctx_len, retriever=None, megatron_tokenizer=None, args=None):

    """CGAP: Prompt a pretrained language model and predict the answer"""
    start_time = time.time()
    print('>>>start CGAP!', flush=True)

    if retriever is None:
        print('no retriever is provided, and we initiate retriever using default configuration')
        retriever, _, args = init_all(args)

    # sample selection
    list_of_prompt_sample_list = []
    for each in [input]:
        query = each['question']
        prompt_sample_list, _ = prompt_sample_selection(query, \
                    args.num_prompt_examples, retriever)
        for i in range(margin_number):
            list_of_prompt_sample_list.append(prompt_sample_list)
    
    input_list=[input] * margin_number

    # context generation, 
    print(">>>Using megatron API to generate the context!")
    context_current_list = context_generation(input_list, list_of_prompt_sample_list, \
                                args.megatron_api_url, megatron_tokenizer, \
                                    args.random_seed, ctx_len, args)
    # answer prediction
    answers_list = answer_prediction(input_list, list_of_prompt_sample_list,\
                            context_current_list, megatron_tokenizer, args)                        
   
    print(">>>finished the genration in {} seconds !".format(\
                time.time()- start_time), flush=True)

    # marjority voting
    final_answer = marginalize_prediction(answers_list)

    result ={
            'question': input['question'],
            'final_answer': final_answer,
            'answer_list': answers_list,
            'generated_context': context_current_list,
    }

    return [result]

def init_models():
    print("> loading tokenizer and encoder", flush=True)
    query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                    'facebook/dpr-question_encoder-multiset-base')
    query_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-multiset-base").cuda()
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                        "facebook/dpr-ctx_encoder-multiset-base")
    ctx_encoder = DPRContextEncoder.from_pretrained(
                    "facebook/dpr-ctx_encoder-multiset-base").cuda()
    return query_tokenizer, query_encoder, ctx_tokenizer, ctx_encoder

def init_others(all_models, args):
    query_tokenizer, query_encoder, ctx_tokenizer, ctx_encoder = all_models
    if args.db_name=='NQ':
        encoded_ctx_files=args.nq_encoded_ctx_file
        prompt_file = args.nq_prompt_file
    else:
        encoded_ctx_files=args.tqa_encoded_ctx_file
        prompt_file=args.tqa_prompt_file

    prompt_data = load_data(prompt_file, with_context=True)
    retriever = MyRetriever(query_encoder,
        query_tokenizer,
        ctx_encoder,
        ctx_tokenizer,
        encoded_ctx_files=encoded_ctx_files,
        data_list=prompt_data,
    )
    print('>>>retriever initialization done!', flush=True)
    return retriever

def init_all(args):
    print('>>>initialize the models', flush=True)
    all_models = init_models()
    print('>>>initialize the retriever', flush=True)
    retriever = init_others(all_models, args)
    query_tokenizer, _, _, _ = all_models

    # this is heuristic, since we don't want to init megatron tokenizer
    if args.length_check:
        # megatron_tokenizer =  get_tokenizer()
        # megatron_tokenizer.pad_token = "<|endoftext|>"
        # assert megatron_tokenizer.tokenize('hello\n\nhello') == [31373, 198, 198, 31373]
        megatron_tokenizer = query_tokenizer
    else:
        megatron_tokenizer = None

    return retriever, megatron_tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='process LIGHT data')
    args = get_tasks_args(parser)
    args.random_seed = random.randrange(1000000)
    assert args.megatron_api_url is not None, 'the megatron api urls is not provided!'

    retriever, megatron_tokenizer = init_all(args)
    print('>>> initialization done!', flush=True)
    margin_number = args.margin_number
    ctx_len = args.ctx_length

    for question in [
        "who is the current president of us?",
    ]:
        input = {"question": question}
        result = cgap(input, margin_number, ctx_len, retriever, megatron_tokenizer, args)
        print(input)
        print(result)
    pass
