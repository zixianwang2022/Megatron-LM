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

from megatron import get_args
from megatron import print_rank_0
from tasks.msdp.metrics import F1Metric
from tqdm import tqdm

import regex
import string
import json
import numpy as np

def evaluate_f1(guess_file, answer_file):
    """Evaluating F1 Score"""

    guess_list = []
    print_rank_0('reading %s' % guess_file)
    with open(guess_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if "<|endoftext|>" in line:
                line = line.replace("<|endoftext|>", "")
            guess_list.append(line)

    answer_list = []
    print_rank_0('reading %s' % answer_file)
    with open(answer_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if line == "no_passages_used":
                line = ""
            answer_list.append(line)

    assert len(guess_list) == len(answer_list), \
        "lengths of guess and answer are different!"

    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list)
    print_rank_0('Precision: %.4f; recall: %.4f; f1: %.4f' % (precision, recall, f1))

    print_rank_0('done :-)')


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

    assert ground_truth_file.endswith('json'), "the ground truth file should be the original json file" 

    prediction_list = []
    print_rank_0('reading %s' % prediction_file)
    with open(prediction_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.replace("Answer: ","")
            line = line.strip()

            if "<|endoftext|>" in line:
                line = line.replace("<|endoftext|>", "")
            prediction_list.append(line)

    ground_truths_list = []
    with open(ground_truth_file, 'r') as f:
        raw_data = json.load(f)
    for each in raw_data:
        ground_truths_list.append(each['answers'])
        
    exactmatch = []
    for i,each in enumerate(prediction_list):
        print("=============")
        print(each)
        print(ground_truths_list[i])
        score = ems(each, ground_truths_list[i])
        exactmatch.append(score)
    
    final_em_score = np.mean(exactmatch)
   
    print_rank_0('Exact Match: %.4f;' % final_em_score)

    print_rank_0('done :-)')

    return final_em_score


def main():
    args = get_args()
    
    evaluate_ems(args.guess_file, args.answer_file)
