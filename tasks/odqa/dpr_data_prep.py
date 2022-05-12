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

"""Model evaluation"""

from tqdm import tqdm

import regex
import string
import json
import numpy as np
from pathlib import Path
import os.path
import torch
import argparse
import random
from tokenizers import SimpleTokenizer
from utils import has_answer

#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def evaluate_ems(prediction_file, ground_truth_file):

    if prediction_file.endswith('withprob.txt'):
        prediction_list, _ = read_prediction_withprob(prediction_file)
    else:
        prediction_list = read_prediction(prediction_file)
    ground_truths_list = []
    
    if ground_truth_file.endswith(('txt', 'lst')):
        raw_data = open(ground_truth_file, 'r')
    else:
        with open(ground_truth_file, 'r') as f:
            raw_data = json.load(f)

    for each in raw_data:
        if ground_truth_file.endswith('txt'):
            each = json.loads(each)
        
        if 'answers' in each:
            ground_truths_list.append(each['answers'])
        elif 'answer' in each:
            ground_truths_list.append(each['answer'])
        else:
            ground_truths_list.append([each])
       
    exactmatch = []

    good_example_list = []
    for i,each in enumerate(prediction_list):
        # print("=============")
        # print(each)
        # print(ground_truths_list[i])
        score = ems(each, ground_truths_list[i])
        # print(score)
        exactmatch.append(score)
        if score:
            good_example_list.append(i)
        
    # good_examples = []
    
    # for i, each in enumerate(raw_data):
    #     if ground_truth_file.endswith('txt'):
    #         each = json.loads(each)
    #     if i in good_example_list:
    #         good_examples.append(each)
    
    # good_examples_save_path = Path(os.path.dirname(ground_truth_file)) / os.path.basename(ground_truth_file).replace('.json', '_good_examples.json')

    # with open(good_examples_save_path, 'w') as f:
    #     json.dump(good_examples, f, indent=4)
    
    # print("write to {} finished!".format(good_examples_save_path))
    
    final_em_score = np.mean(exactmatch)
   
    print('Exact Match: %.4f;' % final_em_score)

    print('done :-)')

    return final_em_score, exactmatch


def read_grounds_truth(ground_truth_file):
    ground_truths_list = []
    
    if ground_truth_file.endswith(('txt', 'lst')):
        raw_data = open(ground_truth_file, 'r')
    else:
        with open(ground_truth_file, 'r') as f:
            raw_data = json.load(f)

    for each in raw_data:
        if ground_truth_file.endswith('txt'):
            each = json.loads(each)
        
        if 'answers' in each:
            ground_truths_list.append(each['answers'])
        elif 'answer' in each:
            ground_truths_list.append(each['answer'])
        else:
            ground_truths_list.append([each])
    
    return ground_truths_list


def read_prediction_withprob(prediction_withprob_file):
    prediction_list = []
    logprob_list = []
    print('reading %s' % prediction_withprob_file)
    with open(prediction_withprob_file, "r") as f:
        for i, line_prob in enumerate(tqdm(f)):
            line_split = line_prob.split('\t')
            logprob = line_split[-1]
            line = "\t".join(line_split[:-1])
            logprob_list.append(float(logprob))

            line = line.replace("Answer:","")
            line = line.replace("Answer: ","")
            line = line.replace('????  ', "")
            line = line.replace('A: ',"")
            line = line.replace("A:", "")
            line = line.strip()

            if "<|endoftext|>" in line:
                line = line.replace("<|endoftext|>", "")
            
            line = normalize_answer(line) # normalize the answer
            prediction_list.append(line)

    return prediction_list, logprob_list

def read_prediction(prediction_file):
    prediction_list = []
    print('reading %s' % prediction_file)
    with open(prediction_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.replace("Answer:","")
            line = line.replace("Answer: ","")
            line = line.replace('????  ', "")
            line = line.replace('A: ',"")
            line = line.replace("A:", "")

            line = line.strip()

            if "<|endoftext|>" in line:
                line = line.replace("<|endoftext|>", "")
            line = normalize_answer(line) # normalize the answer
            prediction_list.append(line)

    return prediction_list


def construct_ideal_data_for_ranker(prediction_file_list, ground_truth_file, context_file_list, save_dpr_file_path=None, n_neg_samples=15, use_golden_answers=True):
    '''this is just for ideal data: each data point [golden context, all the rest 15 * take a context from any other question]'''
    
    # read the predicted answer list
    predicted_answer_list = []
    for i, prediction_file in enumerate(prediction_file_list):
        if prediction_file.endswith('withprob.txt'):
            prediction_list, _ = read_prediction_withprob(prediction_file)
        else:
            prediction_list = read_prediction(prediction_file)
        predicted_answer_list.append(prediction_list)
    # read the context_file_list 
    context_data_list = []
    for i, context_file in enumerate(context_file_list):
        with open(context_file, 'r') as f:
            context_data = f.readlines()
        context_data_list.append(context_data)

    
    filtered_data_save_path = ""
    if save_dpr_file_path is not None:
        filtered_data_save_path = Path(os.path.dirname(save_dpr_file_path)) / os.path.basename(save_dpr_file_path).replace(".json", "-filtered-list.txt")
    # filtered_data_list = filter_pos_neg_list_for_dpr_retriever(prediction_file_list, ground_truth_file, filtered_data_save_path)
    # print("There are {} examples for dpr retriever training~!".format(len(filtered_data_list)))
    filtered_data_list, pos_example_cnt, hard_pos_example_cnt, hard_neg_example_cnt = \
     filter_pos_neg_list_for_dpr_retriever_with_answer_checking(prediction_file_list, ground_truth_file, \
                                                                predicted_answer_list, context_data_list, \
                                                                filtered_data_save_path, \
                                                                use_golden_answers=use_golden_answers)
    print("There are {} examples and {} positive examples, {} hard_positive, {} hard_negative, \
        for dpr retriever training~!".format(len(filtered_data_list), pos_example_cnt, hard_pos_example_cnt, hard_neg_example_cnt))


    # read the original file
    orig_data_list = []
    if ground_truth_file.endswith(('txt', 'lst')):
        raw_data = open(ground_truth_file, 'r')
    else:
        with open(ground_truth_file, 'r') as f:
            raw_data = json.load(f)

    for each in raw_data:
        if ground_truth_file.endswith('txt'):
            each = json.loads(each)
        
        if 'answers' in each:  
            answers = each['answers']
        elif 'answer' in each:
            answers = each['answer']
        
        golden_ctx = each['ctxs'][0]

        orig_data_list.append({"question": each["question"],
                                "answers": answers,
                                "golden_ctxs": golden_ctx,
                                })

    # dpr_data_list = format_ideal_data_for_ranker(filtered_data_list, context_data_list, orig_data_list, n_neg_samples)

    # dpr_data_list = format_ideal_data_for_ranker_new(filtered_data_list, context_data_list, orig_data_list, n_neg_samples)

    dpr_data_list = format_ideal_data_for_ranker_new_new(filtered_data_list, context_data_list, orig_data_list, predicted_answer_list, n_neg_samples)

    if save_dpr_file_path is not None:  
        with open(save_dpr_file_path, "w") as f:
            json.dump(dpr_data_list, f, indent=4)

    return

def format_ideal_data_for_ranker(pos_neg_data_list, context_data_list, orig_data_list, n_neg_samples):

    # construct the data
    dpr_data_list=[]
    for filtered_sample in pos_neg_data_list:
        dpr_data={}
        id = int(filtered_sample["id"])
        
        dpr_data["question"] = orig_data_list[id]["question"]
        dpr_data["answers"] = orig_data_list[id]["answers"]
        dpr_data["positive_ctxs"] = []
        dpr_data["positive_ctxs"].append(orig_data_list[id]["golden_ctxs"])
        dpr_data["negative_ctxs"] = []
        dpr_data["hard_negative_ctxs"] =[]
        # randomly pick 15 context from other samples
        samples_list = random.sample(pos_neg_data_list, n_neg_samples)

        for each in samples_list:
            sample_id = each['id']
            current_list = each["postive_sample_ids"] + each["negative_sample_ids"]
            id = random.sample(current_list,1)
            dpr_data["hard_negative_ctxs"].append(
                {
                    "title": "",
                    "text": context_data_list[id[0]][sample_id].strip().replace("A: ", "")
                }
            )

        dpr_data_list.append(dpr_data)
    
    return dpr_data_list



def format_ideal_data_for_ranker_new(pos_neg_data_list, context_data_list, orig_data_list, n_neg_samples):
    # we will include the negative from the C_gen, and hard_negatives from others.
    # construct the data
    dpr_data_list=[]
    for filtered_sample in pos_neg_data_list:
        dpr_data={}
        id = int(filtered_sample["id"])
        
        dpr_data["question"] = orig_data_list[id]["question"]
        dpr_data["answers"] = orig_data_list[id]["answers"]
        dpr_data["golden_ctxs"] = orig_data_list[id]["golden_ctxs"]
        dpr_data["positive_ctxs"] = []
        dpr_data["negative_ctxs"] = []
        dpr_data["hard_negative_ctxs"] =[]

        pos_list = filtered_sample["postive_sample_ids"]
        neg_list = filtered_sample["negative_sample_ids"]

        for each in pos_list:
            dpr_data["positive_ctxs"].append(
                {
                    "title": "",
                    "text": context_data_list[each][id].strip().replace("A: ", "")
                }
            )

        for each in neg_list:
            dpr_data["negative_ctxs"].append(
                {
                    "title": "",
                    "text": context_data_list[each][id].strip().replace("A: ", "")
                }
            )

        # randomly pick 15 context from other samples, just in case there is not enough "negative_ctxs"
        samples_list = random.sample(pos_neg_data_list, n_neg_samples)

        for each in samples_list:
            sample_id = each['id']
            current_list = each["postive_sample_ids"] + each["negative_sample_ids"]
            id = random.sample(current_list,1)
            dpr_data["hard_negative_ctxs"].append(
                {
                    "title": "",
                    "text": context_data_list[id[0]][sample_id].strip().replace("A: ", "")
                }
            )

        dpr_data_list.append(dpr_data)
    
    return dpr_data_list



def format_ideal_data_for_ranker_new_new(pos_neg_data_list, context_data_list, orig_data_list, predicted_answer_list, n_neg_samples):
    '''we will include the negative/hard_negatives/positives/hard_positives from the C_gen, and other_negatives from others. '''

    # construct the data
    dpr_data_list=[]
    for filtered_sample in pos_neg_data_list:
        dpr_data={}
        id = int(filtered_sample["id"])
        
        dpr_data["question"] = orig_data_list[id]["question"]
        dpr_data["answers"] = orig_data_list[id]["answers"]
        dpr_data["golden_ctxs"] = orig_data_list[id]["golden_ctxs"]
        dpr_data["positive_ctxs"] = []
        dpr_data["negative_ctxs"] = []
        dpr_data["hard_negative_ctxs"] =[]
        dpr_data["hard_positive_ctxs"] = []
        dpr_data["other_negative_ctxs"] = []

        pos_list = filtered_sample["postive_sample_ids"]
        neg_list = filtered_sample["negative_sample_ids"]
        hard_pos_list = filtered_sample["hard_postive_sample_ids"]
        hard_neg_list = filtered_sample["hard_negative_sample_ids"]

        dpr_data["positive_ctxs"] = [
                {"title": "",
                "text": context_data_list[each][id].strip().replace("A: ", ""),
                "predicted_answer": predicted_answer_list[each][id],
                } for each in pos_list ]
        
        dpr_data["negative_ctxs"] = [
                {
                    "title": "",
                    "text": context_data_list[each][id].strip().replace("A: ", ""), 
                    "predicted_answer": predicted_answer_list[each][id],

                } for each in neg_list]
        
        dpr_data["hard_positive_ctxs"] = [
                {"title": "",
                "text": context_data_list[each][id].strip().replace("A: ", ""),
                "predicted_answer": predicted_answer_list[each][id],
                } for each in hard_pos_list ]
        
        dpr_data["hard_negative_ctxs"] = [
                {
                    "title": "",
                    "text": context_data_list[each][id].strip().replace("A: ", ""),
                    "predicted_answer": predicted_answer_list[each][id],
                } for each in hard_neg_list]


        # randomly pick 15 context from other samples, just in case there is not enough "negative_ctxs"
        samples_list = random.sample(pos_neg_data_list, n_neg_samples * 2)

        for each in samples_list:
            sample_id = each['id']
            current_list = each["postive_sample_ids"] + each["negative_sample_ids"]
            if len(current_list) == 0:
                continue
            id = random.sample(current_list,1)
            dpr_data["other_negative_ctxs"].append(
                {
                    "title": "",
                    "text": context_data_list[id[0]][sample_id].strip().replace("A: ", "")
                }
            )
            if len(dpr_data["other_negative_ctxs"]) == n_neg_samples:
                break

        dpr_data_list.append(dpr_data)
    
    return dpr_data_list





def construct_data_for_dpr_retriever(prediction_file_list, ground_truth_file, context_file_list, save_dpr_file_path=None, use_golden_answers=True):

    filtered_data_save_path = ""
    if save_dpr_file_path is not None:
        filtered_data_save_path = Path(os.path.dirname(save_dpr_file_path)) / os.path.basename(save_dpr_file_path).replace(".json", "-filtered-list.txt")
    
    
    # read the context_file_list 
    context_data_list = []
    for i, context_file in enumerate(context_file_list):
        with open(context_file, 'r') as f:
            context_data = f.readlines()
        context_data_list.append(context_data)

    # read the answer file list

    predicted_answer_list = []
    for i, prediction_file in enumerate(prediction_file_list):
        if prediction_file.endswith('withprob.txt'):
            prediction_list, _ = read_prediction_withprob(prediction_file)
        else:
            prediction_list = read_prediction(prediction_file)
        predicted_answer_list.append(prediction_list)

    
    # filtered_data_list = filter_pos_neg_list_for_dpr_retriever(prediction_file_list, ground_truth_file, filtered_data_save_path)
    
    filtered_data_list, pos_example_cnt, hard_pos_example_cnt, hard_neg_example_cnt = \
    filter_pos_neg_list_for_dpr_retriever_with_answer_checking(prediction_file_list, ground_truth_file, \
                                                                predicted_answer_list, context_data_list, \
                                                                filtered_data_save_path, \
                                                                use_golden_answers=use_golden_answers, \
                                                                )
    print("There are {} examples and {} positive examples, {} hard_positive, {} hard_negative, \
        for dpr retriever training~!".format(len(filtered_data_list), pos_example_cnt, hard_pos_example_cnt, hard_neg_example_cnt))


    # read the original file
    orig_data_list = []
    if ground_truth_file.endswith(('txt', 'lst')):
        raw_data = open(ground_truth_file, 'r')
    else:
        with open(ground_truth_file, 'r') as f:
            raw_data = json.load(f)

    for each in raw_data:
        if ground_truth_file.endswith('txt'):
            each = json.loads(each)
        
        if 'answers' in each:  
            answers = each['answers']
        elif 'answer' in each:
            answers = each['answer']
   
        orig_data_list.append({"question": each["question"],
                                "answers": answers,
                                })

    dpr_data_list = format_save_data_for_dpr_retriever(filtered_data_list, context_data_list, orig_data_list, predicted_answer_list)

    if save_dpr_file_path is not None:  
        with open(save_dpr_file_path, "w") as f:
            json.dump(dpr_data_list, f, indent=4)

    return

def format_save_data_for_dpr_retriever(pos_neg_data_list, context_data_list, orig_data_list, predicted_answer_list):

    # construct the data
    dpr_data_list=[]
    for filtered_sample in pos_neg_data_list:
        dpr_data={}
        id = int(filtered_sample["id"])
        pos_list = filtered_sample["postive_sample_ids"]
        neg_list = filtered_sample["negative_sample_ids"]
        
        dpr_data["question"] = orig_data_list[id]["question"]
        dpr_data["answers"] = orig_data_list[id]["answers"]
        dpr_data["positive_ctxs"] = []
        for each in pos_list:
            dpr_data["positive_ctxs"].append(
                {
                    "title": "",
                    "text": context_data_list[each][id].strip().replace("A: ", ""),
                    "predicted_answer": predicted_answer_list[each][id],
                }
            )
        dpr_data["negative_ctxs"] = []
        dpr_data["hard_negative_ctxs"] =[]
        for each in neg_list:
            dpr_data["hard_negative_ctxs"].append(
                {
                    "title": "",
                    "text": context_data_list[each][id].strip().replace("A: ", ""),
                    "predicted_answer": predicted_answer_list[each][id],
                }
            )
        dpr_data_list.append(dpr_data)
    
    return dpr_data_list

def filter_pos_neg_list_for_dpr_retriever(prediction_file_list, ground_truth_file, filtered_data_save_path=None):

    if os.path.exists(filtered_data_save_path) and os.path.getsize(filtered_data_save_path) > 0:
        with open(filtered_data_save_path, 'r') as f:
            filtered_data_list = json.load(f)
        print("Read the filtered data list from file {}".format(filtered_data_save_path))
        return filtered_data_list

    exactmatch_list_list = []
    for i, prediction_file in enumerate(prediction_file_list):
        _, exactmatch_list = evaluate_ems(prediction_file, ground_truth_file)
        # assert len(exactmatch_list) == 79168, 'the length of {} file is not enough'.format(i)
        exactmatch_list_list.append(exactmatch_list)

    filtered_data_list=[]
    
    for j in range(len(exactmatch_list_list[0])):
        pos_list = []
        neg_list = []
        filtered_data={}
        for i in range(len(exactmatch_list_list)):
            if exactmatch_list_list[i][j]:
                pos_list.append(i)
            else:
                neg_list.append(i)

        # if len(pos_list) > 0:
            # valid samples
        
        filtered_data["id"] = j
        filtered_data["postive_sample_ids"] = pos_list
        filtered_data["negative_sample_ids"] = neg_list
        filtered_data_list.append(filtered_data)
    
    if filtered_data_save_path is not None:
        with open(filtered_data_save_path, 'w') as f:
            json.dump(filtered_data_list, f, indent=4)

        print("save the filtered list to file {} done!".format(filtered_data_save_path))


    return filtered_data_list





def filter_pos_neg_list_for_dpr_retriever_with_answer_checking(prediction_file_list, ground_truth_file, \
                                predicted_answer_list, context_data_list,
                                filtered_data_save_path=None, use_golden_answers=True):
    '''the positive context has to be (1) lead to the correct answer (2) the answer was contained in the context'''
    if os.path.exists(filtered_data_save_path) and os.path.getsize(filtered_data_save_path) > 0:
        with open(filtered_data_save_path, 'r') as f:
            filtered_data_list = json.load(f)
        print("Read the filtered data list from file {}".format(filtered_data_save_path))
        return filtered_data_list, 0, 0, 0,

    # read the context_file_list 
    assert context_data_list is not None, 'the context_data_list is empty!'

    # read the original answer file
    check_answers_list = []
    if use_golden_answers:
        print("use the golden answers to filetering!")
        if ground_truth_file.endswith(('txt', 'lst')):
            raw_data = open(ground_truth_file, 'r')
        else:
            with open(ground_truth_file, 'r') as f:
                raw_data = json.load(f)
        for each in raw_data:
            if ground_truth_file.endswith('txt'):
                each = json.loads(each)
            if 'answers' in each:  
                answers = each['answers']
            elif 'answer' in each:
                answers = each['answer']
            check_answers_list.append(answers)
    else:
        # use the predicted answer to filter
        print("use the predicted answer to filter!")
        for answer_list in predicted_answer_list:
            current_list = [[each] for each in answer_list]
            check_answers_list.append(current_list)

    # condition1: answer correct
    exactmatch_list_list = []
    for i, prediction_file in enumerate(prediction_file_list):
        _, exactmatch_list = evaluate_ems(prediction_file, ground_truth_file)
        # assert len(exactmatch_list) == 79168, 'the length of {} file is not enough'.format(i)
        exactmatch_list_list.append(exactmatch_list)

    # condition2: context contains the answer
    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    golden_answer_in_ctx_list_list = []
    if use_golden_answers:
        for i, context_list in enumerate(context_data_list): 
            golden_answer_in_ctx_list_list.append([])
            for j, (check_answers, context) in enumerate(zip(check_answers_list, context_list)):
                golden_answer_in_ctx_list_list[i].append(has_answer(check_answers, context, tokenizer, match_type='string'))
    else:
        for i, (context_list, check_answers) in enumerate(zip(context_data_list, check_answers_list)): 
            golden_answer_in_ctx_list_list.append([])
            for j, (each_check_answers, context) in enumerate(zip(check_answers, context_list)):
                golden_answer_in_ctx_list_list[i].append(has_answer(each_check_answers, context, tokenizer, match_type='string'))


    filtered_data_list=[]
    data_num = len(exactmatch_list_list[0])
    context_num = len(exactmatch_list_list)

    pos_example_cnt = 0 
    hard_pos_example_cnt = 0
    hard_neg_example_cnt = 0
    neg_example_cnt = 0

    for j in range(data_num):
        pos_list = []
        hard_pos_list = [] # do not contain the answer, but lead to correct answer
        neg_list = []
        hard_neg_list = [] # contain the answer, but lead to an incorrect answer.
        filtered_data={}
        for i in range(context_num):
            if exactmatch_list_list[i][j] and golden_answer_in_ctx_list_list[i][j]:
                pos_list.append(i)
            elif exactmatch_list_list[i][j] is True and golden_answer_in_ctx_list_list[i][j] is False:
                hard_pos_list.append(i)
            elif exactmatch_list_list[i][j] is False and golden_answer_in_ctx_list_list[i][j] is True:
                hard_neg_list.append(i)
            else:
                neg_list.append(i)

        if len(pos_list) > 0:
            pos_example_cnt += 1
        
        # if len(pos_list) == 0 and len(hard_pos_list) > 0:
        #     hard_pos_example_cnt += 1
        
        # if len(neg_list) == 0 and len(hard_neg_list) > 0:
        #     hard_neg_example_cnt += 1

        if len(hard_pos_list) > 0:
            hard_pos_example_cnt += 1
        
        if len(hard_neg_list) > 0:
            hard_neg_example_cnt += 1
        
        if len(neg_list) > 0:
            

        
        filtered_data["id"] = j
        filtered_data["postive_sample_ids"] = pos_list
        filtered_data["negative_sample_ids"] = neg_list
        filtered_data["hard_postive_sample_ids"] = hard_pos_list
        filtered_data["hard_negative_sample_ids"] = hard_neg_list

        filtered_data_list.append(filtered_data)
    
    if filtered_data_save_path is not None:
        with open(filtered_data_save_path, 'w') as f:
            json.dump(filtered_data_list, f, indent=4)

        print("save the filtered list to file {} done!".format(filtered_data_save_path))

    return filtered_data_list, pos_example_cnt, hard_pos_example_cnt, hard_neg_example_cnt


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    # parameters for the open-domain QA
    group.add_argument("--guess-file", type=str, default=None,
                       help='the predicted answer file')
    group.add_argument("--answer-file", type=str, default=None,
                       help='the file with golden answer')

    group.add_argument('--save-context-path', type=str, default="",
                       help='the path to save the generated context files')

    group.add_argument('--save-context-prompt-path', type=str, default="",
                       help='the path to save the generated context files')

    group.add_argument('--save-similarity-file-path', type=str, default="",
                       help='the path to save the generated context files')

    group.add_argument('--save-dpr-file-path', type=str, default="",
                       help='the path to save the generated context files')

    group.add_argument('--n-neg-samples', type=int, default=1,
                       help='negative numbers')

    group.add_argument('--use-golden-answers', default=False, action='store_true',)


    return parser.parse_args()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='process argument')
    args = get_tasks_args(parser)
    guess_file_list = args.guess_file.strip(',').split(',')
    context_file_list=args.save_context_path.strip(',').split(',')

    # construct_data_for_dpr_retriever(guess_file_list, args.answer_file, context_file_list, args.save_dpr_file_path, args.use_golden_answers)

    construct_ideal_data_for_ranker(guess_file_list, args.answer_file, context_file_list, \
                                     args.save_dpr_file_path, args.n_neg_samples, args.use_golden_answers)
    
    # topk_context(guess_file_list, context_file_list, topk=2)
    # result_analysis(args.guess_file, args.answer_file, args.save_context_path)

    # calculate the similarity score between the generated context and the retrieved golden
    # similarity_score(args.compare_file)
