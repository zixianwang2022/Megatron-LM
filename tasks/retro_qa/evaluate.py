import regex
from tqdm import tqdm
import numpy as np
import string
import json

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
            if prediction_file.endswith("jsonl"):
                line = json.loads(line)["pred"]
                # print(line)
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

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def evaluate_ems(prediction_file, ground_truth_file, dev_num=3000):
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
    if "dev" in ground_truth_file:
        raw_data = raw_data[:dev_num]
        prediction_list = prediction_list[:dev_num]

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

    final_em_score = np.mean(exactmatch)

    print('Exact Match: %.4f;' % final_em_score)

    print('done :-)')

    return final_em_score, exactmatch


def eval_1_3b():

    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/dev.json"
    for step in range(1500, 16500, 1500):
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_bert_retriever_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_dev_greedy_0_3000_8_{}.txt"
        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_bert_retriever_ctx1_1.3b_32_3e-6_0.0/generate_1.3b_dev_greedy_0_3000_{}.txt"
        p_file = prediction_file.format(step)
        # evaluate_ems(p_file, ground_truth_file)

    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_bert_retriever_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_test_greedy_0_4000_8_15000.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_bert_retriever_ctx1_1.3b_32_3e-6_0.0/generate_1.3b_test_greedy_0_4000_15000.txt"
    evaluate_ems(prediction_file, ground_truth_file)


if __name__ == "__main__":
    # NQ
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting/generate_843m_test_greedy_0_400_194000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting/generate_843m_test_greedy_0_400_195312.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_2_375000_1_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532_short_format_1.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc/generate_2b_test_greedy_0_400_1417624.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting/generate_2b_test_greedy_0_400_97656.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting/generate_2b_test_greedy_0_400_97656.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc/generate_8b_test_greedy_0_400_1417624.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting/generate_8b_test_greedy_0_400_97656.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/generate_8b_test_greedy_0_400_97656.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc/generate_22b_test_greedy_0_400_708812.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting/generate_22b_test_greedy_0_400_48828.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/generate_22b_test_greedy_0_400_48828.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1/generate_43b_test_greedy_0_400_472541.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1/generate_43b_test_greedy_0_400_32552.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_nq_43b_test_greedy_0_400_32552.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_32000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_31000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_30000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_29000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_28000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_27000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)


    # TQA
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA/test.json"
    #
    # ## 800M
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr/generate_tqa_843m_test_greedy_0_1100_2835248.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting/generate_tqa_843m_test_greedy_0_1100_194000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting/generate_tqa_843m_test_greedy_0_1100_195312.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # ## 2B
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc/generate_tqa_2b_test_greedy_0_1100_1417624.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting/generate_tqa_2b_test_greedy_0_1100_97656.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting/generate_tqa_2b_test_greedy_0_1100_97656.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # ## 8B
    #
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc/generate_tqa_8b_test_greedy_0_1100_1417624.concat.txt.period.txt"
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting/generate_tqa_8b_test_greedy_0_1100_97656.concat.txt.period.txt"
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/generate_tqa_8b_test_greedy_0_1100_97656.concat.txt.period.txt"
    evaluate_ems(prediction_file, ground_truth_file)
    #
    # ## 22B
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc/generate_tqa_22b_test_greedy_0_1100_708812.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting/generate_tqa_22b_test_greedy_0_1100_48828.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/generate_tqa_22b_test_greedy_0_550_48828.concat.txt.period.txt"
    evaluate_ems(prediction_file, ground_truth_file)

    # ## 43B

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1/generate_tqa_43b_test_greedy_0_550_472541.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1/generate_tqa_43b_test_greedy_0_550_32552.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_tqa_43b_test_greedy_0_400_32000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)


