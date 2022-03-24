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
from pathlib import Path
import os.path
import torch
from collections import Counter

def perplexity():
    import math
    from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
# Load pre-trained model (weights)
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def score(sentence):
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss=model(tensor_input, lm_labels=tensor_input)
        return math.exp(loss)


def similarity_score(compare_file):

    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
    from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder

    query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                    'facebook/dpr-question_encoder-multiset-base')
    query_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-multiset-base").cuda()
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                        "facebook/dpr-ctx_encoder-multiset-base")
    ctx_encoder = DPRContextEncoder.from_pretrained(
                    "facebook/dpr-ctx_encoder-multiset-base").cuda()

    with open(compare_file, 'r') as f:
        r_g_data =  json.load(f)

    score_list = []
    for each in r_g_data:
        re_text = each['golden_ctx']['title'] + ' ' + each['golden_ctx']['text']
        # re_text = each['question']
        ge_text = each['gen_ctx']
        # ge_text = each['question']

        with torch.no_grad():
            # get the query embeddings
            re_ids = query_tokenizer.encode(re_text, truncation=True, max_length=512)
            re_ids = torch.LongTensor([re_ids]).cuda()
            re_emb = query_encoder(input_ids=re_ids).pooler_output
            re_emb = re_emb[0]

            ge_ids = ctx_tokenizer.encode(ge_text, truncation=True, max_length=512)
            ge_ids = torch.LongTensor([ge_ids]).cuda()
            ge_emb = ctx_encoder(input_ids=ge_ids).pooler_output
            ge_emb = ge_emb[0]

            similarity_score = re_emb.matmul(ge_emb)
            similarity_score = similarity_score.tolist()

            print(similarity_score)
            score_list.append(similarity_score)

    score = np.mean(score_list)
    print('The similarity score avarage is {}'.format(score))

    return score



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

    prediction_list = []
    print_rank_0('reading %s' % prediction_file)
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
            prediction_list.append(line)

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
        print("=============")
        print(each)
        print(ground_truths_list[i])
        score = ems(each, ground_truths_list[i])
        print(score)
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
   
    print_rank_0('Exact Match: %.4f;' % final_em_score)

    print_rank_0('done :-)')

    return final_em_score, exactmatch


def marginalize_prediction(prediction_file_list):
    prediction_list_list = []
    for prediction_file in prediction_file_list:
        prediction_list = []
        print_rank_0('reading %s' % prediction_file)
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
        prediction_list_list.append(prediction_list)
    
    most_predicted_answer_list = []
    for j in range(len(prediction_list_list[0])):
        current_list=[]
        for i in range(len(prediction_list_list)):
            current_list.append(prediction_list_list[i][j])
        x = Counter(current_list)
        (most_predicted_answer, _) = x.most_common()[0]
        most_predicted_answer_list.append(most_predicted_answer)
    
    return most_predicted_answer_list


def evaluate_ems_multiple_marginalize(prediction_file_list, ground_truth_file):

    # assert ground_truth_file.endswith('json'), "the ground truth file should be the original json file" 
    
    most_predicted_answer_list = marginalize_prediction(prediction_file_list)


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

    exactmatch=[]
    for i,each in enumerate(most_predicted_answer_list):
        print("=============")
        print(each)
        print(ground_truths_list[i])
        score = ems(each, ground_truths_list[i])
        print(score)
        exactmatch.append(score)


    final_em_score = np.mean(exactmatch)
   
    print_rank_0('Final Exact Match: %.4f;' % final_em_score)

    print_rank_0('done :-)')

    return


def evaluate_ems_multiple(prediction_file_list, ground_truth_file):

    # assert ground_truth_file.endswith('json'), "the ground truth file should be the original json file" 
    exactmatch_list_list = []
    for prediction_file in prediction_file_list:
        _, exactmatch_list = evaluate_ems(prediction_file, ground_truth_file)
        exactmatch_list_list.append(exactmatch_list)

    new_exactmatch_list=[]
    for j in range(len(exactmatch_list_list[0])):
        sum = 0
        for i in range(len(exactmatch_list_list)):
            sum += int(exactmatch_list_list[i][j])
        new_exactmatch_list.append(bool(sum))
    
    new_score = np.mean(new_exactmatch_list)

    print_rank_0('Final Exact Match: %.4f;' % new_score)

    print_rank_0('done :-)')

    return new_score






def result_analysis(prediction_file, ground_truth_file, gen_ctx_file):

    # assert ground_truth_file.endswith('json'), "the ground truth file should be the original json file" 

    gen_ctx_list = []

    with open(gen_ctx_file, 'r') as f:
        gen_ctx_list = f.readlines()

    prediction_list = []
    print_rank_0('reading %s' % prediction_file)
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
            prediction_list.append(line)

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
    
    true_list = []
    false_list = []

    exactmatch = []
    for i,each in enumerate(prediction_list):
        print("=============")
        print(each)
        print(ground_truths_list[i])
        score = ems(each, ground_truths_list[i])
        print(score)
        exactmatch.append(score)
        if score:
            true_list.append(i)
        else:
            false_list.append(i)
    
    #
    analysis_true_data = []
    analysis_false_data = []
    all_data = []
    analysis_data = {}
    for i in range(len(prediction_list)):
        analysis_data = {}
        analysis_data['question'] = raw_data[i]['question']
        analysis_data['golden_ctx'] = raw_data[i]['ctxs'][0]
        if 'answer' in raw_data[i]: 
            analysis_data['golden_ans'] = raw_data[i]['answer'] 
        elif 'answers' in raw_data[i]:
           analysis_data['golden_ans'] = raw_data[i]['answers']
        else:  
            analysis_data['golden_ans'] = raw_data[i]['target'] 
        analysis_data['gen_ctx'] = gen_ctx_list[i]
        analysis_data['gen_ans'] = prediction_list[i]

        all_data.append(analysis_data)

        if i < 10:
            print(analysis_data)

        if i in true_list:
            analysis_true_data.append(analysis_data)
        else:
            analysis_false_data.append(analysis_data) 
    
    save_result_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.txt.all')
    with open(save_result_path, 'w') as f:
        json.dump(all_data, f, indent=4)

    save_true_result_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.txt.true')
    save_false_result_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.txt.false')


    with open(save_true_result_path, 'w') as f:
        json.dump(analysis_true_data, f, indent=4)
    print("save the true file done!")
    with open(save_false_result_path, 'w') as f:
        json.dump(analysis_false_data, f, indent=4)
    print("save the false file done!")

    save_true_list_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.true_list.txt')
    save_false_list_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.false_list.txt')

    with open(save_true_list_path, 'w') as f:
        for each in true_list:
            f.write("%s\n" % each)

    with open(save_false_list_path, 'w') as f:
        for each in false_list:
            f.write("%s\n" % each)

    print("save the true list and false_list to file done!")

    final_em_score = np.mean(exactmatch)
   
    print_rank_0('Exact Match: %.4f;' % final_em_score)

    print_rank_0('done :-)')

    return final_em_score


def main():
    args = get_args()
    
    # evaluate_ems(args.guess_file, args.answer_file)
    guess_file_list = args.guess_file.strip(',').split(',')
    # evaluate_ems_multiple(guess_file_list, args.answer_file)
    evaluate_ems_multiple_marginalize(guess_file_list, args.answer_file)

    # result_analysis(args.guess_file, args.answer_file, args.save_context_path)

    # calculate the similarity score between the generated context and the retrieved golden

    # similarity_score(args.compare_file)
