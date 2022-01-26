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
from megatron.text_generation import generate_and_post_process
from .data import load_data
import random

def call_model_api(inputs, tokens_to_generate):
    """Calling the model api to get the output generations"""
    
    args = get_args()

    # The following is an example of using the Megatron API
    # You can also implement your own API function to place this part
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {"prompts": [inputs], "tokens_to_generate": tokens_to_generate, "top_k": 1}
    data_json = json.dumps(data)
    outputs = requests.put(args.megatron_api_url, headers=headers, data=data_json).json()["text"][0]

    input_len = len(inputs)
    outputs = outputs[input_len:]
    outputs = outputs.split("\n")[0].strip()
    
    return outputs


def read_prompts(prompt_path, prompt_type, n_example):
    """Read prompt data"""

    if prompt_type == "knowledge":
        # prompts for the knowledge generation
        prompt_examples_dict = {}
        # read prompt_path
        with open(prompt_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                line_dict = json.loads(line)
                key = list(line_dict.keys())[0]
                
                if key not in prompt_examples_dict:
                    prompt_examples = line_dict[key]
                    prompt = ""
                    for instance in prompt_examples:
                        instance = instance.strip()
                        prompt += instance + " \n"
                    prompt_examples_dict[key] = prompt

        return prompt_examples_dict

    else:
        # prompts for the response generation
        # read prompt_path
        prompt = ""
        with open(prompt_path, "r") as f:
            prompt_examples = f.readlines()
            prompt_examples = prompt_examples[:n_example]
            for instance in prompt_examples:
                instance = instance.strip()
                prompt += instance + " \n"

        return prompt


def generate_samples_by_calling_api():
    """ Generate outputs by calling"""
    args = get_args()
    assert args.prompt_type in ["knowledge", "response"], \
                "Please input a correct prompt type!"

    if args.prompt_type == "knowledge":
        # read knowledge generation prompts
        knwl_gen_prompt_dict = read_prompts(
            args.prompt_file, args.prompt_type, args.num_prompt_examples)
        
    else:
        resp_gen_prompt = read_prompts(
            args.prompt_file, args.prompt_type, args.num_prompt_examples)

    # read the test data
    fname = open(args.input_file, "r")
    test_sample_list = fname.readlines()
    # create output file
    fname_out = open(args.output_file, "w")

    # call the api to get the output generations
    for test_sample in test_sample_list:
        test_sample = test_sample.strip()
        splits = test_sample.split("\t")
        topic = splits[0]

        # prepare the inputs for the api
        if args.prompt_type == "knowledge":
            ## inputs = prompt + current test
            # get the prompt
            turns = splits[1].split(" [SEP] ")
            last_turn = turns[-1]
            key = topic + " " + last_turn
            inputs = knwl_gen_prompt_dict[key]

            # add current test
            inputs += "( " + last_turn + " ) " + topic + " =>"

        else:
            # inputs = prompt + current test
            # get the prompt
            inputs = resp_gen_prompt

            # add current test
            turns = splits[1].split(" [SEP] ")
            knowledge = splits[2]
            last_turn = turns[-1]
            last_turn = " ".join(word_tokenize(last_turn))
            knowledge = " ".join(word_tokenize(knowledge))
            knowledge = knowledge.strip()
            last_turn = last_turn.strip()
            inputs += "Topic: " + topic + ". "
            inputs += "User says: " + last_turn + " "
            inputs += "We know that: " + knowledge + " "
            inputs += "System replies:"

        # get the output generations from the api, 
        # and write to the output file
        generations = call_model_api(inputs, args.out_seq_length)
        fname_out.write(generations)
        fname_out.write("\n")

    fname.close()
    fname_out.close()


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


def prompt_sample_selection(data_list, query = "", k=10, is_random=True):
    # currently only random selection
    if is_random:
        return random.sample(data_list, k)
    else: 
        ## option1: return the top
        return NotImplemented
    return data[0]

def generate_samples_by_prompting_input_from_file(model):
    """Prompt a pretrained language model to generate answer"""
    
    # get tokenizer
    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        # load the data from input and prompt file
        raw_data = load_data(args.input_file)
        prompt_data = load_data(args.prompt_file)
        input_count = len(raw_data)
        
        if args.output_file is None:
            output_file = args.input_file + ".out"
            print('`output-file` not specified, setting '
                    'it to {}'.format(output_file))
        else:
            output_file = args.output_file

        fname_out = open(output_file, "w")

    input_pos = 0
    model.eval()
    # perform prompting
    with torch.no_grad():
        while True:
            raw_text_len = 0
            if mpu.is_pipeline_first_stage() \
               and mpu.get_tensor_model_parallel_rank() == 0:
                input = raw_data[input_pos]
                propmt_question = 'Question: ' + input['question'] + ' ?'
                prompt_sample_list= prompt_sample_selection(prompt_data, input['question'], args.num_prompt_examples)

                prompt_text = ''
                for each in prompt_sample_list:
                    prompt_text += 'Question: ' + each['question'] + '? ' + 'Answer: ' + each['answers'][0] + '\n'
                
                prompt_text += propmt_question
                input_pos += 1
                raw_text_len = len(prompt_text)
            
            else:
                prompt_text = "EMPTY TEXT"

            if input_pos % 100 == 0:
                print_rank_0("input_pos: %d" % input_pos)

            # print("Prompt text is {} \n".format(prompt_text))
            # print('==============================')
            outputs = generate_and_post_process(
                        model=model, 
                        prompts=[prompt_text], 
                        tokens_to_generate=args.out_seq_length,
                        top_k_sampling=1)
            prompts_plus_generations = outputs[0]
            prompts_plus_generations = prompts_plus_generations[0]

            # write the generated output to the output file
            if mpu.get_tensor_model_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():

                    generations = prompts_plus_generations[raw_text_len:]
                    generations = generations.split("\n")[0]
                    generations = generations.strip()
                    fname_out.write(generations)
                    fname_out.write("\n")

            raw_text = None
            if input_pos == input_count:
                return


def main():

    args = get_args()
    if args.api_prompt:
        # obtain the generations by calling the api
        generate_samples_by_calling_api()
        return

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
    generate_samples_by_prompting_input_from_file(model)
