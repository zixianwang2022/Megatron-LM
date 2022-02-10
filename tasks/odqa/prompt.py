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
from .data import load_data, load_data_distributed, load_piQA_data
from .utils import write_output
import random
import os.path
from pathlib import Path
import shutil

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

    # zero-shot
    if k==0:
        return []

    # few-shot: currently only random selection
    if is_random:
        return random.sample(data_list, k)
    else: 
        ## option1: return the top
        return NotImplemented
    return data[0]

def post_process_generations(generations, min_token_length=5, sep='\n'):
    # return the first string that has length longer than 5
    # print("=====================")
    # print(generations)
    generations_split = generations.split(sep)
    for each in generations_split:
        if len(each.strip()) >= min_token_length:
            return each.strip()
    
    return "No proper answer!"


def construct_input_prompt(input_list, prompt_data, format='', num_prompt_examples=0):

    prompt_text_list = []
    raw_text_len_list = []

    for input in input_list:
        propmt_question = ''
        prompt_sample_list= prompt_sample_selection(prompt_data, input['question'], num_prompt_examples)
        
        # Option1: GPT-3 paper format
        if format == 'GPT-3':
            propmt_question = 'Q: ' + input['question'] + '?\n' + 'A: '   # for NaturalQuestions
            # propmt_question = 'Q: ' + input['question'] + '\n' + 'A: '  # for TriviaQA and WebQuestions
            
            prompt_text = ''
            for each in prompt_sample_list:
                answer=''
                if 'target' in each:
                    answer = each['target']
                else:
                    answer = each['answer'][0]
                prompt_text += 'Q: ' + each['question'] + '?\n' + 'A: ' + answer + '\n' # for NaturalQuestions
                # prompt_text += 'Q: ' + each['question'] + '\n' + 'A: ' + each['target'] + '\n'  # for TriviaQA and WebQuestions
        
        # option2: EleutherAI format
        elif format == 'Eleuther-AI':
            propmt_question = 'Q: ' + input['question'] + '\n\n' + 'A: '   # for NaturalQuestions
            # propmt_question = 'Question: ' + input['question'] + '\n' + 'Answer: ' # for TriviaQA and WebQuestions

            prompt_text=''
            for each in prompt_sample_list:
                answer=''
                if 'target' in each:
                    answer = each['target']
                else:
                    answer = each['answer'][0]                

                prompt_text  += 'Q: ' + input['question'] + '\n\n' + 'A: ' + answer + '\n'  # for NaturalQuestions
                # prompt_text += 'Question: ' + input['question'] + '\n' + 'Answer: ' + answer + '\n' # for TriviaQA and WebQuestions
        
        # Option3: Ours
        else: 
            if num_prompt_examples == 0:
                propmt_question = 'Question: ' + input['question'] + '\n' + 'Answer: '  

            else:
                propmt_question = 'Question: ' + input['question'] + '\n'

            prompt_text = ''
            for each in prompt_sample_list:
                answer=''
                if 'target' in each:
                    answer = each['target']
                else:
                    answer = each['answer'][0]
                
                prompt_text += 'Question: ' + each['question'] + '\n' + 'Answer: ' + answer + '\n'

        prompt_text += propmt_question
        prompt_text_list.append(prompt_text)
        raw_text_len = len(prompt_text)
        raw_text_len_list.append(raw_text_len)
    
    return prompt_text_list, raw_text_len_list


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
                propmt_question = 'Question: ' + input['question'] + '? ' 
                # propmt_question = 'Question: ' + input['question'] + '? '  + 'Answer: '

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
                return True
    


def batch_generate_samples_by_prompting_input_from_file(model):
    """Prompt a pretrained language model to generate answer"""
    
    # get tokenizer
    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage():
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
            print("output_file is {}".format(output_file))
    
    input_pos = 0
    bz = args.micro_batch_size
    model.eval()
    # perform prompting
    with torch.no_grad():
        with open(output_file, "w") as fname_out:
            while True:
                print("input_pos is {} and input_count is {}, and rank is {}".format(input_pos, \
                    input_count, torch.distributed.get_rank()), flush=True)      

                if mpu.is_pipeline_first_stage() \
                   and mpu.get_tensor_model_parallel_rank() == 0:
                    start_pos = input_pos
                    end_pos = input_pos + bz if input_pos + bz < input_count else input_count
                    input_list = raw_data[start_pos: end_pos]

                    # to peng: you can change the third parameters from 'GPT-3', 'Eleuther-AI', or 'Nvidia', to indicate
                    prompt_text_list, raw_text_len_list = construct_input_prompt(input_list, prompt_data, args.prompt_format, args.num_prompt_examples)

                    input_pos += len(prompt_text_list)
                    
                    if input_pos % 100 == 0:
                        print_rank_0("input_pos: {}".format(input_pos))

                outputs = generate_and_post_process(
                            model=model, 
                            prompts=prompt_text_list, 
                            tokens_to_generate=args.out_seq_length,
                            top_k_sampling=1)
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
                    print("Rank {} finished the genration!".format(torch.distributed.get_rank()), flush=True)
                    return     

# just for PiQA
def batch_generate_samples_by_prompting_input_from_file_for_piQA(model):
    """Prompt a pretrained language model to generate answer"""
    
    # get tokenizer
    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage():
        # load the data from input and prompt file
        raw_data = load_piQA_data(args.input_file)
        prompt_data = load_piQA_data(args.prompt_file)
        input_count = len(raw_data)

        if args.output_file is None:
            output_file = args.input_file + ".out"
            print('`output-file` not specified, setting '
                    'it to {}'.format(output_file))
        else:
            output_file = args.output_file
            print("output_file is {}".format(output_file))
    
    input_pos = 0
    bz = args.micro_batch_size
    model.eval()
    # perform prompting
    with torch.no_grad():
        with open(output_file, "w") as fname_out:
            while True:
                print("input_pos is {} and input_count is {}, and rank is {}".format(input_pos, \
                    input_count, torch.distributed.get_rank()), flush=True)      

                if mpu.is_pipeline_first_stage() \
                   and mpu.get_tensor_model_parallel_rank() == 0:
                    start_pos = input_pos
                    end_pos = input_pos + bz if input_pos + bz < input_count else input_count
                    input_list = raw_data[start_pos: end_pos]
                    prompt_text_list_1 = []
                    prompt_text_list_2 = []
                    for input in input_list:
                        propmt_question_1, propmt_question_2  = '', ''
                        if args.num_prompt_examples == 0:
                            # GPT-3 style
                            # propmt_question_1 = input['goal'] + ' ' + input['sol1']  
                            # propmt_question_2 = input['goal'] + ' ' + input['sol2'] 
                            # GPT-Neo style
                            propmt_question_1 = 'Question: ' + input['goal'] + '\nAnswer: ' + input['sol1']  
                            propmt_question_2 = 'Question: ' + input['goal'] + '\nAnswer: ' + input['sol2'] 


                        prompt_sample_list= prompt_sample_selection(prompt_data, input['goal'], args.num_prompt_examples)
                        prompt_text = ''
                        for each in prompt_sample_list:
                            if int(each['golden']) == 0:
                                #GPT-3
                                # prompt_text += each['goal'] + ' ' + each['sol2'] + ' '
                                # GPT-Neo
                                prompt_text += 'Question: ' + each['goal'] + '\nAnswer: ' + each['sol1'] + '\n\n'

                            elif int(each['golden']) == 1:
                                # prompt_text += each['goal'] + ' ' + each['sol2'] + ' '
                                prompt_text += 'Question: ' + each['goal'] + '\nAnswer: ' + each['sol2'] + '\n\n'

                        prompt_text_1 = prompt_text + propmt_question_1
                        prompt_text_2 = prompt_text + propmt_question_2
                        prompt_text_list_1.append(prompt_text_1)
                        prompt_text_list_2.append(prompt_text_2)

                        input_pos += 1
                    
                    if input_pos % 100 == 0:
                        print_rank_0("rank is {}, input_pos: {}".format(torch.distributed.get_rank(),input_pos))

                outputs_1 = generate_and_post_process(
                            model=model, 
                            prompts=prompt_text_list_1, 
                            tokens_to_generate=args.out_seq_length,
                            return_output_log_probs=True,
                            top_k_sampling=args.top_k_sampling,
                            top_p_sampling=args.top_p_sampling,
                            temperature = args.temperature)
                output_log_probs_1 = outputs_1[2]
                
                outputs_2 = generate_and_post_process(
                            model=model, 
                            prompts=prompt_text_list_2, 
                            tokens_to_generate=args.out_seq_length,
                            return_output_log_probs=True,
                            top_k_sampling=args.top_k_sampling,
                            top_p_sampling=args.top_p_sampling,
                            temperature = args.temperature)

                output_log_probs_2 = outputs_2[2]


                # write the generated output to the output file
                if mpu.get_tensor_model_parallel_rank() == 0:
                    if mpu.is_pipeline_first_stage():
                        for log_prob1, log_prob2 in zip(output_log_probs_1, output_log_probs_2):
                            avg_log_prob1 = sum(log_prob1) / len(log_prob1)
                            avg_log_prob2 = sum(log_prob2) / len(log_prob2)
                            # print("The two probability is {} and {}".format(avg_log_prob1, avg_log_prob2))
                            if avg_log_prob1 >= avg_log_prob2:
                                predicted_lable = '0'
                            else:
                                predicted_lable = '1'
                            fname_out.write(predicted_lable)
                            fname_out.write("\n")
                
                if input_pos == input_count:
                    print("Rank {} finished the genration!".format(torch.distributed.get_rank()), flush=True)
                    return 
    


def distributed_generate_samples_by_prompting_input_from_file(model):
    """Prompt a pretrained language model to generate answer"""
    
    # get tokenizer
    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage():
        # load the data from input and prompt file
        # raw_data = load_data_distributed(args.input_file, mpu.get_tensor_model_parallel_rank(), mpu.get_tensor_model_parallel_world_size())
        raw_data = load_data_distributed(args.input_file, torch.distributed.get_rank(), torch.distributed.get_world_size())
        prompt_data = load_data(args.prompt_file)
        input_count = len(raw_data)

        if args.output_file is None:
            output_file = args.input_file + ".out"
            print('`output-file` not specified, setting '
                    'it to {}'.format(output_file))
        else:
            out_dir = Path(os.path.dirname(args.output_file)) / args.exp_name / 'tmp_results'
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / ('%d.txt'%torch.distributed.get_rank())
            print("output_file is {}".format(output_file))
    
    input_pos = 0
    model.eval()
    # perform prompting
    with torch.no_grad():
        with open(output_file, "w") as fname_out:
            # while True:
            while input_pos < input_count:
                print("input_pos is {} and input_count is {}, and rank is {}".format(input_pos, \
                     input_count, torch.distributed.get_rank()), flush=True)      

                raw_text_len = 0
                if mpu.is_pipeline_first_stage() \
                   and mpu.get_tensor_model_parallel_rank() == 0:
                        input = raw_data[input_pos]
                        print("Question is {}".format(input['question']))
                        # propmt_question = 'Question: ' + input['question'] + '? '
                        propmt_question = ''
                        propmt_question = 'Question: ' + input['question'] + '? '  + 'Answer: '
                        prompt_sample_list= prompt_sample_selection(prompt_data, input['question'], args.num_prompt_examples)
                        prompt_text = ''
                        for each in prompt_sample_list:
                            if 'target' in each:
                                prompt_text += 'Question: ' + each['question'] + '? ' + 'Answer: ' + each['target'] + '\n'
                            else:
                                prompt_text += 'Question: ' + each['question'] + '? ' + 'Answer: ' + each['answers'][0] + '\n'
                        
                        prompt_text += propmt_question
                        input_pos += 1
                        raw_text_len = len(prompt_text)
                        
                        if input_pos % 100 == 0:
                            print_rank_0("rank is {}, input_pos: {}".format(torch.distributed.get_rank(),input_pos))
                ###
                

                if mpu.get_tensor_model_parallel_rank() == 0:
                    outputs = generate_and_post_process(
                                model=model, 
                                prompts=[prompt_text, prompt_text, prompt_text], 
                                tokens_to_generate=args.out_seq_length,
                                top_k_sampling=1)
                    prompts_plus_generations = outputs[0]
                    # prompts_plus_generations = prompts_plus_generations[0]
                print("================")
                print(len(prompts_plus_generations))
                print("the rank is {} and generation is {}".format(torch.distributed.get_rank(), prompts_plus_generations), flush=True)
                # print(prompts_plus_generations, flush=True)

                # # write the generated output to the output file
                # if mpu.get_tensor_model_parallel_rank() == 0:
                #     if mpu.is_pipeline_first_stage():
                #         generations = prompts_plus_generations[raw_text_len:].strip()
                #         generations_str = post_process_generations(generations, min_token_length=5, sep='\n')
                #         fname_out.write(str(input['id']) + '\t' + generations_str)
                #         fname_out.write("\n")
                prompt_text=None
                if input_pos == input_count:
                    print("Rank {} finished the genration!".format(torch.distributed.get_rank()), flush=True)
                    return 

                else:
                    continue
    


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
    # generate_samples_by_prompting_input_from_file(model)
    # batch_generate_samples_by_prompting_input_from_file(model)

    # for PIQA, need to merge with other functions later
    batch_generate_samples_by_prompting_input_from_file_for_piQA(model)


    # the distrubted generation
    # out_dir = Path(os.path.dirname(args.output_file)) / args.exp_name / 'tmp_results'
    # if torch.distributed.get_rank() in [0, -1]: # remove the existing failed results
    #     if out_dir.exists(): 
    #         print("remove the existing directory {}!".format(out_dir), flush=True)
    #         shutil.rmtree(out_dir)
    
    # distributed_generate_samples_by_prompting_input_from_file(model)
    
    # if torch.distributed.get_rank() in [0, -1]:
    #     print("Rank {} starting writing to the outputfile!".format(torch.distributed.get_rank()), flush=True)
    #     out_dir = Path(os.path.dirname(args.output_file)) / args.exp_name / 'tmp_results'
    #     write_output(out_dir, args.output_file)
    
    # return "Done"

