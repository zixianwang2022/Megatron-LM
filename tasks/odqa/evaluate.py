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
from collections import Counter
import torch
from .sampling import sample
# from .tokenizers import SimpleTokenizer
from .utils import has_answer


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




def similarity_score(query_text, context_text_list, topk, query_tokenizer, \
                    query_encoder, ctx_tokenizer, ctx_encoder):

    with torch.no_grad():
        # get the query embeddings
        query_ids = query_tokenizer.encode(query_text, truncation=True, max_length=512)
        query_ids = torch.LongTensor([query_ids]).cuda()
        query_emb = query_encoder(input_ids=query_ids).pooler_output
        query_emb = query_emb[0]
        
        context_emb_list=[]
        for idx, context_text in enumerate(context_text_list):        
            context_ids = ctx_tokenizer.encode(context_text, truncation=True, max_length=512)
            context_ids = torch.LongTensor([context_ids]).cuda()
            context_emb = ctx_encoder(input_ids=context_ids).pooler_output

            context_emb_list = torch.cat((context_emb_list, context_emb), \
                dim=0) if idx > 0 else context_emb


        similarity_score_list = context_emb_list.matmul(query_emb)

        # also rank according to the score
        scores, indices = torch.topk(similarity_score_list, k=topk)
        scores = scores.tolist()
        indices = indices.tolist()
        similarity_score_list = similarity_score_list.tolist()

    return similarity_score_list, scores, indices


def similarity_score_upper(golden_context, context_text_list, topk, ctx_tokenizer, ctx_encoder):

    with torch.no_grad():
        # get the golden_context embeddings
        query_ids = ctx_tokenizer.encode(golden_context, truncation=True, max_length=512)
        query_ids = torch.LongTensor([query_ids]).cuda()
        query_emb = ctx_encoder(input_ids=query_ids).pooler_output
        query_emb = query_emb[0]
        
        context_emb_list=[]
        for idx, context_text in enumerate(context_text_list):        
            context_ids = ctx_tokenizer.encode(context_text, truncation=True, max_length=512)
            context_ids = torch.LongTensor([context_ids]).cuda()
            context_emb = ctx_encoder(input_ids=context_ids).pooler_output

            context_emb_list = torch.cat((context_emb_list, context_emb), \
                dim=0) if idx > 0 else context_emb


        similarity_score_list = context_emb_list.matmul(query_emb)

        # also rank according to the score
        scores, indices = torch.topk(similarity_score_list, k=topk)
        scores = scores.tolist()
        indices = indices.tolist()
        similarity_score_list = similarity_score_list.tolist()

    return similarity_score_list, scores, indices




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
    for i, prediction_file in enumerate(prediction_file_list):
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



def marginalize_prediction_with_answer_filtering(prediction_file_list, context_file_list):
    prediction_list_list = []
    for i, prediction_file in enumerate(prediction_file_list):
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
    data_num = len(prediction_list_list[0])
    ctx_num = len(prediction_list_list)

    # read the context file 
    context_data_list = []
    for i, context_file in enumerate(context_file_list):
        with open(context_file, 'r') as f:
            context_data = f.readlines()
        context_data_list.append(context_data)

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    for j in range(data_num):
        current_list_with_answer_filtering=[]
        current_list =[]
        for i in range(ctx_num):
            current_predicted_answer = prediction_list_list[i][j]
            if has_answer([current_predicted_answer], context_data_list[i][j], tokenizer, match_type='string'):
                current_list_with_answer_filtering.append(current_predicted_answer)
            current_list.append(current_predicted_answer)   
        if len(current_list_with_answer_filtering) > 0:    
            x = Counter(current_list_with_answer_filtering)
        else:
            x = Counter(current_list)
        # x=Counter(current_list)
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



def evaluate_ems_multiple_marginalize_with_answer_filtering(prediction_file_list, context_file_list, ground_truth_file):

    # assert ground_truth_file.endswith('json'), "the ground truth file should be the original json file" 
    
    # most_predicted_answer_list = marginalize_prediction(prediction_file_list)
    most_predicted_answer_list = marginalize_prediction_with_answer_filtering(prediction_file_list,\
                                context_file_list)


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
    print_rank_0('reading %s' % prediction_withprob_file)
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

    return prediction_list


def beam_prediction(prediction_file_list, beam_size=4):
    prediction_list_list = []
    logprob_list_list = []
    for i, prediction_file in enumerate(prediction_file_list):
        prediction_list, logprob_list = read_prediction_withprob(prediction_file)
        prediction_list_list.append(prediction_list)
        logprob_list_list.append(logprob_list)
    
    beam_predicted_answer_list_list = []
    beam_predicted_prob_list=[]
    beam_predicted_logprob_list=[]
    answer_topk_list=[]

    l1=len(prediction_list_list[0])
    for j in range(l1):
        current_list=[]
        l= len(prediction_list_list)
        for i in range(l):
            current_list.append(prediction_list_list[i][j])
        x = Counter(current_list)
        beam_answer_list = x.most_common()[:beam_size]

        answer_list=[]
        count_list=[]
        logprob_list=[]
        answer_topk={}

        for k, (answer, count) in enumerate(beam_answer_list):
            count_list.append(count)
            answer_list.append(answer)
            logprob = 0.0
            answer_topk[k+1]=[]
            for i in range(l):
                if prediction_list_list[i][j] == answer:
                    logprob += logprob_list_list[i][j]
                    answer_topk[k+1].append(i)
            avg_logprob = logprob / count
            logprob_list.append(avg_logprob)
        
        answer_topk_list.append(answer_topk)

        count_sum=sum(count_list)
        prob_list = [x/count_sum for x in count_list]
        # print("====="*10)

        # print(logprob_list)
        logprob_list = my_softmax(np.array(logprob_list))
        logprob_list = logprob_list.tolist()
        assert abs(sum(logprob_list) - 1.0) <= 0.000001
        # print(logprob_list)

        # padding
        if len(prob_list) < beam_size:
            diff = beam_size - len(prob_list)
            prob_list = prob_list + diff * [0.0]
            answer_list = answer_list + [""] * diff
            logprob_list = logprob_list + diff * [0.0]

        # print(answer_list)
        # print(prob_list)
        # print(logprob_list)

        beam_predicted_answer_list_list.append(answer_list)
        beam_predicted_prob_list.append(prob_list)
        beam_predicted_logprob_list.append(logprob_list)

    return beam_predicted_answer_list_list, beam_predicted_prob_list, \
        beam_predicted_logprob_list, answer_topk_list

def my_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def evaluate_ems_beam(prediction_file_list, ground_truth_file):

    # step 1: answer ranking
    beam_size = 2

    beam_predicted_answer_list_list, beam_predicted_prob_list, \
        beam_predicted_logprob_list,  = beam_prediction(prediction_file_list, beam_size=beam_size)

    # step 2: answer sampling
    new_answer_list = []
    for each_prob, answer_list in zip(beam_predicted_prob_list, beam_predicted_answer_list_list):
        beam_logits = torch.tensor(data=[each_prob], dtype=torch.float32,\
                                    device=torch.cuda.current_device()) 
        beam_logits_new = sample(beam_logits, top_k=1, top_p=0.0, temperature=1.0, vocab_size=beam_size)
        print("====")
        print(beam_logits)
        print(beam_logits_new)

        index = beam_logits_new.tolist()[0]
        new_answer_list.append(answer_list[index])

    # step 3: calculate the final ems
    ground_truths_list = read_grounds_truth(ground_truth_file)

    exactmatch_list=[]
    for i,each in enumerate(new_answer_list):
        score = 0
        # for each_prediction in each:
        score += int(ems(each, ground_truths_list[i]))
        exactmatch_list.append(bool(score))
    
    new_score = np.mean(exactmatch_list)
   
    print_rank_0('Final Exact Match: %.4f;' % new_score)

    print_rank_0('done :-)')
    return new_score


def evaluate_ems_cgen_answer_ranking(prediction_file_list, ground_truth_file):

    # step 1: get the C_gen ranked scores

    args = get_args()
    assert args.save_similarity_file_path is not None
    cgen_similarity_list, topk_scores_list, topk_indices_list = rank_c_gen([], ground_truth_file, topk=1, similarity_list_file=args.save_similarity_file_path)

    prediction_list_list = []
    for i, prediction_file in enumerate(prediction_file_list):
        prediction_list = read_prediction(prediction_file)
        prediction_list_list.append(prediction_list)


    # step 2: get the answer probability

    # prediction_list_list = []
    # logprob_list_list = []
    # for i, prediction_file in enumerate(prediction_file_list):
    #     prediction_list, logprob_list = read_prediction_withprob(prediction_file)
    #     prediction_list_list.append(prediction_list)
    #     logprob_list_list.append(logprob_list)

    # beam_size=2
    # beam_predicted_answer_list_list, beam_predicted_prob_list, \
    #     beam_predicted_logprob_list, answer_topk_list = beam_prediction(prediction_file_list, beam_size=beam_size)

    # step 3: combining them together. 
    #         Option 1: P(c_gen|x)*P(y|c_gen, x) \
    #         Option 2: Score(C_gen, x) + Score (answer | C_gen, x)
    #         Option 3: considering only the top-ranked answers for option 1

    # option 1:

    # answer_score_list_list =[]
    # for logprob_list in logprob_list_list:
    #     logprob_list = my_softmax(np.array(logprob_list))
    #     logprob_list = logprob_list.tolist()
    #     assert abs(sum(logprob_list) - 1.0) <= 0.000001
    #     answer_score_list_list.append(logprob_list)

    # combined_scores_list = []
    # for c_gen_scores, answer_scores in zip(cgen_similarity_list, answer_score_list_list):
    #     combined_scores = [c_gen_score / 100 + answer_score for c_gen_score, answer_score in zip(c_gen_scores, answer_scores)]
    #     # combined_scores = [c_gen_score * answer_score for c_gen_score, answer_score in zip(c_gen_scores, answer_scores)]
 
    #     combined_scores_list.append(combined_scores)
    
    top1_answer_list=[]
    # (1) this is the combined scores, seems not that good
    # for i, (topk_dict, combined_scores) in enumerate(zip(answer_topk_list, combined_scores_list)):
    #     print("====")
    #     topk_scores_list = []
    #     for topk_ids in topk_dict.keys():
    #         score = 0.0
    #         for each in topk_dict[topk_ids]:
    #             print("id is {} and top {}".format(each, topk_ids))
    #             score += combined_scores[int(each)]
    #         score = score / len(topk_dict[topk_ids])
    #         topk_scores_list.append(score)
        
    #     top1_index = np.argmax(np.array(topk_scores_list))
    #     print("new top1 index is {}".format(top1_index))
    #     top1_answer = beam_predicted_answer_list_list[i][top1_index]
    #     top1_answer_list.append(top1_answer)

    # (2) use the C_gen score from the top-ranked answers 
    # for i, (topk_dict, c_gen_scores) in enumerate(zip(answer_topk_list, cgen_similarity_list)):
    #     print("====")
    #     topk_scores_list = []
    #     for topk_ids in topk_dict.keys():
    #         score = 0.0
    #         for each in topk_dict[topk_ids]:
    #             print("id is {} and top {}".format(each, topk_ids))
    #             score += c_gen_scores[int(each)]
    #         score = score / len(topk_dict[topk_ids])
    #         topk_scores_list.append(score)
        
    #     top1_index = np.argmax(np.array(topk_scores_list))
    #     print("new top1 index is {}".format(top1_index))
    #     top1_answer = beam_predicted_answer_list_list[i][top1_index]
    #     top1_answer_list.append(top1_answer)

    #(3) use the top1-answer from the combined score.
    # for i, combined_scores in enumerate(combined_scores_list):
    #     print("====")
    #     top1_index = np.argmax(np.array(combined_scores))
    #     print("new top1 index is {}".format(top1_index))
    #     top1_answer = prediction_list_list[top1_index][i]
    #     top1_answer_list.append(top1_answer)

    # (4) this is using the top ranked C_gen's answer for voting
    topk=13
    new_prediction_list_list=[]
    import random
    for i, c_gen_scores in enumerate(cgen_similarity_list):
        new_topk_prediction_list=[]
        topk_scores, topk_indices = torch.topk(torch.tensor(c_gen_scores, dtype=torch.float32), k=topk)
        topk_indices = topk_indices.tolist()
        # let us test random 
        # topk_indices = random.choices([*range(0,16,1)], k=topk)
        new_topk_prediction_list = [prediction_list_list[indices][i] for indices in topk_indices]
        assert len(new_topk_prediction_list) == topk
        new_prediction_list_list.append(new_topk_prediction_list)
    
    most_predicted_answer_list = []
    for current_list in new_prediction_list_list:
        x = Counter(current_list)
        (most_predicted_answer, _) = x.most_common()[0]
        most_predicted_answer_list.append(most_predicted_answer)

        
    top1_answer_list=most_predicted_answer_list
 


    # (4) this is the marginize answer
    # for each in beam_predicted_answer_list_list:
    #     top1_answer_list.append(each[0])
    
    # (5) this is the answer reranking marginized answer
    # for answer_list, logprob_list in zip(beam_predicted_answer_list_list, beam_predicted_logprob_list):
    #     print("====")
    #     print(logprob_list)
    #     index = np.argmax(np.array(logprob_list))
    #     print(index)
    #     pred_answer = answer_list[index]
    #     top1_answer_list.append(pred_answer)

    # step 4: calculate the final ems
    ground_truths_list = read_grounds_truth(ground_truth_file)

    exactmatch_list=[]
    for i,each in enumerate(top1_answer_list):
        score = 0
        # for each_prediction in each:
        score += int(ems(each, ground_truths_list[i]))
        exactmatch_list.append(bool(score))
    
    new_score = np.mean(exactmatch_list)
   
    print_rank_0('Final Exact Match: %.4f;' % new_score)

    print_rank_0('done :-)')
    return new_score



def topk_read(prediction_file_list, beam_size=4):
    '''read the topk answers'''
    prediction_list_list = []
    for i, prediction_file in enumerate(prediction_file_list):
        prediction_list = read_prediction(prediction_file)
        prediction_list_list.append(prediction_list)
    
    beam_predicted_answer_list_list = []
    beam_predicted_prob_list=[]
    answer_topk_list=[]

    l1=len(prediction_list_list[0])
    for j in range(l1):
        current_list=[]
        l= len(prediction_list_list)
        for i in range(l):
            current_list.append(prediction_list_list[i][j])
        x = Counter(current_list)
        beam_answer_list = x.most_common()[:beam_size]

        answer_list=[]
        count_list=[]
        answer_topk={}
        
        for k, (answer, count) in enumerate(beam_answer_list):
            count_list.append(count)
            answer_list.append(answer)
            answer_topk[k+1]=[]
            for i in range(l):
                if prediction_list_list[i][j] == answer:
                    answer_topk[k+1].append(i)
        
        answer_topk_list.append(answer_topk)

        count_sum=sum(count_list)
        prob_list = [x/count_sum for x in count_list]
        # print("====="*10)

        # padding
        if len(prob_list) < beam_size:
            diff = beam_size - len(prob_list)
            prob_list = prob_list + diff * [0.0]
            answer_list = answer_list + [""] * diff

        beam_predicted_answer_list_list.append(answer_list)
        beam_predicted_prob_list.append(prob_list)

    return beam_predicted_answer_list_list, beam_predicted_prob_list, answer_topk_list



def topk_context(prediction_file_list, context_file_list, topk=1):
    '''get the top-k context based on the answer frequency'''
    import random
    import re
    beam_size=3

    assert topk <= beam_size
    _, _, answer_topk_list = topk_read(prediction_file_list, beam_size=beam_size)

    # read the context file list
    context_data_list = []
    for context_file in context_file_list:
        with open(context_file, 'r') as f:
            context_data = f.readlines()
        context_data_list.append(context_data)

    new_topk_context_file_path = Path(os.path.dirname(context_file_list[-1])) / re.sub('_rnd[0-9]+.txt', '.top{}.txt'.format(topk), os.path.basename(context_file_list[-1]))

    new_topk_context_list=[]
    for i, each in enumerate(answer_topk_list):
        new_topk_context = ""
        for top_k in each.keys():
            if top_k > topk:
                break
            top_k_id_list = each[top_k]
            top_k_id = random.choice(top_k_id_list)
            new_topk_context += context_data_list[top_k_id][i].strip() + "\t"
        new_topk_context_list.append(new_topk_context)
    
    with open(new_topk_context_file_path, 'w') as f:
        for new_topk_context in new_topk_context_list:
            f.write(new_topk_context)
            f.write('\n')

    print("write to {} done! :P".format(new_topk_context_file_path))

    return

def rank_c_gen(context_file_list, ground_truth_file, topk=1, similarity_list_file=None):
    '''ranking the generated context by similarity score between the query and C_gen'''
    similarity_list, topk_scores_list, topk_indices_list = [], [], []
    if os.path.exists(similarity_list_file) and os.path.getsize(similarity_list_file) > 0:
        if similarity_list_file.endswith('csv'):
            import csv
            print("read the context score from the csv file!")
            with open(similarity_list_file, 'r') as fin:
                wr = csv.reader(fin)
                similarity_list = list(wr)
            
            print(similarity_list[0])
            print(len(similarity_list))

            similarity_list_new=[]
            for similarity in similarity_list:
                similarity_list_new.append([float(each) for each in similarity])

            for each in similarity_list_new:
                topk_scores, topk_indices = torch.topk(torch.tensor(each, dtype=torch.float32), k=topk)
                topk_scores_list.append(topk_scores.tolist())
                topk_indices_list.append(topk_indices.tolist())

            return similarity_list_new, topk_scores_list, topk_indices_list

        elif similarity_list_file.endswith('txt'):
            print("read the context score from the {} file!".format(similarity_list_file))
            with open(similarity_list_file, 'r') as f:
                topk_indices_list = f.readlines()
            return [], [], topk_indices_list

    
    print("calculate the context score!")
    # read the groundtruth file 
    ground_truths_list=[]   
    if ground_truth_file.endswith(('txt', 'lst')):
        raw_data = open(ground_truth_file, 'r')
    else:
        with open(ground_truth_file, 'r') as f:
            raw_data = json.load(f)
    for each in raw_data:
        if ground_truth_file.endswith('txt'):
            each = json.loads(each)
        ground_truths_list.append(each)
    
    # read the context file list
    context_data_list = []
    for context_file in context_file_list:
        with open(context_file, 'r') as f:
            context_data = f.readlines()
        context_data_list.append(context_data)

    # initiate the retriever
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRConfig, BertConfig
    from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
    
    query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-multiset-base')
    # query_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base").cuda()
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                        "facebook/dpr-ctx_encoder-multiset-base")
    # ctx_encoder = DPRContextEncoder.from_pretrained(
    #                 "facebook/dpr-ctx_encoder-multiset-base").cuda()

    # ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    ctx_encoder = DPRContextEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
    # ctx_encoder.from_pretrained('/gpfs/fs1/projects/gpu_adlr/datasets/dasu/dpr/models_convert_to_huggingface/ctx_encoder/')
    # ctx_encoder.from_pretrained('/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/models_convert_to_huggingface/ctx_encoder_tqa_multiset_39/')
    ctx_encoder.from_pretrained('/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/models_convert_to_huggingface/ctx_encoder_tqa_39/')


    ctx_encoder.cuda()

    # query_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    query_encoder = DPRQuestionEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
    # query_encoder.from_pretrained('/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/models_convert_to_huggingface/question_encoder_tqa_multiset_39/')
    query_encoder.from_pretrained('/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/models_convert_to_huggingface/question_encoder_tqa_39/')

    query_encoder.cuda()
    

    similarity_list=[]
    topk_scores_list=[]
    topk_indices_list=[]

    file_num = len(context_data_list)
    data_num = len(context_data_list[0])

    # calculate the similarity
    for i in range(data_num):
        current_query = ground_truths_list[i]["question"]
        current_golden_ctx = ground_truths_list[i]["ctxs"][0]["title"] + " " + ground_truths_list[i]["ctxs"][0]["text"]
        current_context_list=[]
        for j in range(file_num):
            current_context_list.append(context_data_list[j][i])
        
        current_similarity_score_list, current_topk_scores, current_topk_indices =  similarity_score(current_query, \
            current_context_list, topk,\
            query_tokenizer, \
            query_encoder, \
            ctx_tokenizer, \
            ctx_encoder)

        # current_similarity_score_list, current_topk_scores, current_topk_indices =  similarity_score_upper(current_golden_ctx, \
        #     current_context_list, topk,\
        #     ctx_tokenizer, \
        #     ctx_encoder)

        similarity_list.append(current_similarity_score_list)
        topk_scores_list.append(current_topk_scores)
        topk_indices_list.append(current_topk_indices)

    return similarity_list, topk_scores_list, topk_indices_list

def evaluate_topk_cgen_ems(prediction_file_list, context_file_list, ground_truth_file, topk=1):
    '''evaluate the answer ems generated by the topk C_gen'''
    args = get_args()

    similarity_list, topk_scores_list, topk_indices_list = rank_c_gen(context_file_list, \
        ground_truth_file, topk=1, similarity_list_file=args.save_similarity_file_path)

    # save the similarity list since it takes a long time to compute
    if args.save_similarity_file_path is not None and os.path.exists(args.save_similarity_file_path) is False:
        import csv
        with open(args.save_similarity_file_path, 'w') as fout:
            wr = csv.writer(fout)
            wr.writerows(similarity_list)
        print("write the similarity list to file {} done!".format(args.save_similarity_file_path))


    ground_truths_list = read_grounds_truth(ground_truth_file)

    prediction_answer_list = []
    for i, prediction_file in enumerate(prediction_file_list):
        # prediction_list, _ = read_prediction_withprob(prediction_file)
        prediction_list =  read_prediction(prediction_file)
        prediction_answer_list.append(prediction_list)

    exactmatch=[]
    # random pick one
    from random import randrange
    import random
    # topk_indices_list = [[randrange(16)] for each in range(len(ground_truths_list))]
    # get the top1 answer from the top1 C_gen
    assert len(topk_indices_list) == len(ground_truths_list), 'the length of the topk is not equal to ground truth'
    for i,topk_id_str in enumerate(topk_indices_list):
        # top1_id = topk_id[0]
        topk_ids = topk_id_str.strip(',').split(',')
        top1_id = random.choice(topk_ids)
        print(top1_id)
        top1_answer = prediction_answer_list[int(top1_id)][i]
        
        score = ems(top1_answer, ground_truths_list[i])
        print("=====")
        print(top1_answer)
        print(ground_truths_list[i])
        print(score)
        exactmatch.append(score)

    # calcluate ems
    top1_cgen_ems_score = np.mean(exactmatch)
    print_rank_0('Exact Match: %.4f;' % top1_cgen_ems_score)

    print_rank_0('done :-)')

    return top1_cgen_ems_score


def main():
    args = get_args()
    
    evaluate_ems(args.guess_file, args.answer_file)
    # guess_file_list = args.guess_file.strip(',').split(',')
    # evaluate_ems_multiple(guess_file_list, args.answer_file)
    # evaluate_ems_multiple_marginalize(guess_file_list, args.answer_file)
    # evaluate_ems_beam(guess_file_list, args.answer_file)

    # context_file_list=args.save_context_path.strip(',').split(',')
    # topk_context(guess_file_list, context_file_list, topk=2)
    # result_analysis(args.guess_file, args.answer_file, args.save_context_path)

    # calculate the similarity score between the generated context and the retrieved golden

    # similarity_score(args.compare_file)
    # evaluate_topk_cgen_ems(guess_file_list, context_file_list, args.answer_file, topk=1)
    # evaluate_ems_cgen_answer_ranking(guess_file_list, args.answer_file,)

    # this is not useful.
    # evaluate_ems_multiple_marginalize_with_answer_filtering(guess_file_list, context_file_list, args.answer_file)
