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
import requests
import random
import os.path
import time
import argparse



def load_data(data_path=None, with_context=False):
    assert data_path
    if data_path.endswith(('.jsonl', 'txt')):
        data = open(data_path, 'r')
    elif data_path.endswith(('.json')):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    
    examples = []
        
    for k, example in enumerate(data):
        if data_path is not None and data_path.endswith(('.jsonl', 'txt')):
            example = json.loads(example)
        new_example = {}
        new_example['id'] = k
        new_example['question'] = example['question']
        if 'answers' in example:
            new_example['answers'] = example['answers']
        elif 'answer' in example:
            new_example['answers'] = example['answer']
        if 'target' in example:
            new_example['target'] = example['target']
        if with_context:
            if 'ctxs' in example:
                new_example['ctxs'] = example['ctxs'][0]
            else:
                new_example['ctxs'] = 'no context'
        examples.append(new_example)

    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()
    return examples


def call_model_api(inputs, tokens_to_generate, top_k_sampling,\
                                    top_p_sampling,temperature, random_seed, url):
    """Calling the model api to get the output generations"""
    

    # The following is an example of using the Megatron API
    # You can also implement your own API function to place this part
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {"prompts": inputs, \
            "tokens_to_generate": tokens_to_generate, \
            "top_k_sampling": 0, \
            "top_p_sampling": top_p_sampling, \
            "temperature": temperature, \
            "random_seed": random_seed, \
            }
    data_json = json.dumps(data)
    outputs = requests.put(url, headers=headers, data=data_json).json()["text"]

    # input_len = len(inputs)
    # outputs = outputs[input_len:]
    # outputs = outputs.split("\n")[0].strip()
    
    return outputs


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

def post_process_generations(generations, min_token_length=5, sep='\n'):
    # return the first string that has length longer than 5
    generations_split = generations.split(sep)
    start_pos, end_pos = 0,0
    for each in generations_split:
        if len(each.strip()) >= min_token_length + len('Answer: '):
            end_pos += len(each)
            return each.strip(), (start_pos, end_pos)
        else:
            start_pos += len(each) + len(sep)
            end_pos += len(each) + len(sep)

    
    return "No proper answer!", (0, 0)


def context_generation_by_call_api(input_list, args):

    # return the generated context
    context_prompt_list = []
    context_prompt_len_list=[]
    generation_list = []

    with open(args.save_context_prompt_path, 'r') as f:
        data = f.readlines()
        for input in input_list:
            id = input['id']
            context_prompt_dict = json.loads(data[id])
            context_prompt_list.append(context_prompt_dict[str(id)])
            # if id<5:
            #     print("======")
            #     print(context_prompt_dict[str(id)])
            context_prompt_len_list.append(len(context_prompt_dict[str(id)]))

    assert args.save_context_path is not None, 'the save_context_path should not be None'
    
    context_prompt_batch = context_prompt_list

    print("using the megatron API to generate!")
    outputs_batch = call_model_api(
                        inputs=context_prompt_batch, 
                        tokens_to_generate=100,
                        top_k_sampling=0,
                        top_p_sampling=0.9,
                        temperature = args.temperature,
                        random_seed=args.random_seed,
                        url=args.megatron_api_url,
                        )

    prompts_plus_generations_list = outputs_batch

    for prompts_plus_generations, raw_text_len in zip(prompts_plus_generations_list, context_prompt_len_list):
        generations = prompts_plus_generations[raw_text_len:].strip()
        generations_str, _ = post_process_generations(generations, min_token_length=5, sep='\n')
        generation_list.append(generations_str)
    
    return generation_list
 


def step2_batch_generate_context_by_call_api(args):
    import csv 
    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    raw_data = load_data(args.input_file, args.with_context)
    raw_data = raw_data[48+1465:]
    input_count = len(raw_data)

    input_pos = 0
    bz = args.micro_batch_size
    start_time = time.time()
    
    fcontxt_out = open(args.save_context_path, "a")
    writer_ = csv.writer(fcontxt_out, delimiter=',')
    writer_.writerow(['id', 'api_response'])

    # perform prompting
    while True:
        print("input_pos is {} and input_count is {}".format(input_pos, \
            input_count))      

        start_pos = input_pos
        end_pos = input_pos + bz if input_pos + bz < input_count else input_count
        input_list = raw_data[start_pos: end_pos]
        context_current_list=[]
        context_current_list= context_generation_by_call_api(input_list, args)
        
        if input_pos < int(args.micro_batch_size) * 5:
            print("======generated context samples=====!")
            print(context_current_list[0].encode('utf-8'))   
                     
        for i, context_generation in enumerate(context_current_list):
            writer_.writerow(
                [str(i+start_pos), \
                    context_generation]
                )

        input_pos += len(context_current_list)

        if input_pos % 100 == 0:
            print("input_pos: {}".format(input_pos))
            
        if input_pos == input_count:
            print("finished the context genration in {} seconds !".format(\
                time.time()- start_time))
            break    

    return


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    # parameters for the open-domain QA
    group.add_argument("--input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument('--prompt-file', type=str, default=None,
                       help='prompting file')
    group.add_argument('--num-prompt-examples', type=int, default=10,
                       help='number of prompt examples')
    group.add_argument('--out-seq-length', type=int, default=100,
                       help='output sequence length')
    group.add_argument('--api-prompt', default=False, action="store_true",
                       help='setup model api for prompting')
    group.add_argument('--megatron-api-url', type=str, default=None,
                       help='url of the megatron api')
    group.add_argument('--top-p-sampling', type=float, default=0.0,
                       help='the top-p value')
    group.add_argument('--top-k-sampling', type=float, default=0.0,
                       help='the top-k value')
    group.add_argument('--temperature', type=float, default=0.0,
                       help='the temperature value')
    group.add_argument('--micro-batch-size', type=int, default=1,
                       help='the batch_size')

    group.add_argument('--save-context-path', type=str, default="",
                       help='the path to save the generated context files')
    group.add_argument('--is-context-generated', default=False, action="store_true",
                       help='whether generated the context or use retreival')
    group.add_argument('--with-context', default=False, action="store_true",
                       help='weather will append the context in the prompt construction')
    group.add_argument('--random-seed', default=-1, type=int,
                       help='the random seed that megatron model used to generate text')


    group.add_argument('--save-context-prompt-path', type=str, default="",
                       help='the path to save the generated context files')



    return parser.parse_args()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='process LIGHT data')
    args = get_tasks_args(parser)

    
    if args.api_prompt:
        step2_batch_generate_context_by_call_api(args)
