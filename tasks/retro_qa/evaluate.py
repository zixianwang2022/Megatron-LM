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

    # eval_1_3b()
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_1.3b_32_3e-6_0.0_4/generate_1.3b_test_beam_0_4000.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_8.3b_32_3e-6_0.0_8/generate_8.3b_test_greedy_0_5000.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_8.3b_32_2e-6_0.0_8/generate_8.3b_test_greedy_0_4000.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_20.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_8.3b_32_1e-6_0.0_8_bk/generate_8.3b_test_greedy_0_6000_16.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_title_8.3b_32_1e-6_0.0_8_r192/generate_8.3b_test_greedy_0_6000_8_15000.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_title_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_6000.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_6000_10500.txt"
    # prediction_file = "/home/pengx/projects/retro/checkpoints/applications/nq_retro_ft_126m_32_1e-5_0.0_8/generate_126m_test_greedy_0_4000.txt"
    # prediction_file = "/home/pengx/projects/retro/checkpoints/applications/nq_retro_ft_126m_32_1e-5_0.0_16/generate_126m_test_greedy_0_4000.txt"
    # prediction_file = "/home/pengx/projects/retro/checkpoints/applications/nq_retro_ft_126m_32_1e-5_0.0_20/generate_126m_test_greedy_0_4000.txt"
    # prediction_file = "/home/pengx/projects/retro/checkpoints/applications/nq_retro_ft_357m_32_1e-5_0.0_20/generate_357m_test_greedy_0_4000.txt"
    # prediction_file = "/home/pengx/projects/retro/checkpoints/applications/nq_retro_ft_357m_32_1e-5_0.0_16/generate_357m_test_greedy_0_4000.txt"
    # prediction_file = "/mnt/fsx-main/pengx/projects/retro/checkpoints/applications/nq_retro_ft_357m_32_1e-5_0.0_20/generate_357m_test_beam_0_4000.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_title_8.3b_32_1e-6_0.0_8_r192/generate_8.3b_test_greedy_0_6000_8_{}.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_with_title_ctx_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_6000_{}.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_8.3b_32_1e-6_0.0/generate_without_ctx/generate_8.3b_test_greedy_0_6000_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_6000_10500.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_10500.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_4000_10500.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_4000_8_10500.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx0_1.3b_32_3e-6_0.0/generate_1.3b_test_greedy_0_4000_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx0_357m_32_1e-5_0.0/generate_357m_test_greedy_0_4000_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx0_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_4000_15000.txt"
    prediction_file = "prediction_NQ.jsonl"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_357m_32_1e-5_0.0_2/generate_357m_test_greedy_0_4000_2_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_357m_32_1e-5_0.0_4/generate_357m_test_greedy_0_4000_4_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_357m_32_1e-5_0.0_8/generate_357m_test_greedy_0_4000_8_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_357m_32_1e-5_0.0_16/generate_357m_test_greedy_0_4000_16_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/bf16_without_title_nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_4000_8_10500.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/bf16_nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_4000_8_10500.txt"

    # wbx test
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/benz_dpr_finetuned/test.json"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"

    ## 8.3B
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_8.3b_test_greedy_0_4000_389532.txt.bak"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_8.3b_test_greedy_0_4000_389532.txt.bak.truncate10.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_full_8_375000_retro.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_full_8_375000_retro.txt.truncate20.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/qa_retro/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla//generate_8.3b_test_greedy_0_4000_8_340.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/qa_retro/nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_dev_greedy_0_4000_8_10500.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_test_greedy_0_4000_8_15000.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_test_greedy_0_4000_8_10500.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_nq_8.3b_test_greedy_0_400_389532_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)


    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_nq_8.3b_test_greedy_0_400_389532_short_format_1.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_nq_8.3b_test_greedy_0_400_389532_short_format_2.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_nq_8.3b_test_greedy_0_400_389532_short_format_3.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)


    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_2_375000_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_1_375000_1_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_1_375000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_2_375000_1_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_2_375000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_3_375000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_4_375000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_5_375000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_6_375000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_nq_8.3b_test_greedy_0_400_2_80000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_nq_8.3b_test_greedy_0_400_2_60000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_nq_8.3b_test_greedy_0_400_2_50000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_nq_8.3b_test_greedy_0_400_2_50000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_nq_8.3b_test_greedy_0_400_2_100000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_nq_8.3b_test_greedy_0_400_2_100000_2_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr/generate_843m_test_greedy_0_400_2835248.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting/generate_843m_test_greedy_0_400_194000.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting/generate_843m_test_greedy_0_400_195312.concat.txt.period.txt"
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_2_375000_1_short_format.concat.txt.period.txt"
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532_short_format_1.concat.txt.period.txt"
    evaluate_ems(prediction_file, ground_truth_file)

    # ## 1.3B
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532.concat.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532.concat.txt.truncate32.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532.concat.txt.truncate20.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532.concat.txt.truncate10.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_8_375000.concat.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_8_375000.concat.txt.truncate32.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_8_375000.concat.txt.truncate20.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_8_375000.concat.txt.truncate10.txt"
    # evaluate_ems(prediction_file, ground_truth_file)



    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA/test.json"
    ## 8.3B
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_8.3b_test_greedy_concat_1100_389532.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_8.3b_test_greedy_concat_1100_389532.txt.truncate10.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_concatenated_8_375000.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_concatenated_8_375000.txt.truncate10.txt"

    ## 1.3B
    # prediction_file = "//lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_tqa_1.3b_test_greedy_0_1100_389532.concat.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "//lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_tqa_1.3b_test_greedy_0_1100_389532.concat.txt.truncate32.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "//lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_tqa_1.3b_test_greedy_0_1100_389532.concat.txt.truncate20.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "//lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_tqa_1.3b_test_greedy_0_1100_389532.concat.txt.truncate10.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_tqa_1.3b_test_greedy_0_1100_8_375000.catcat.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_tqa_1.3b_test_greedy_0_1100_8_375000.catcat.txt.truncate32.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_tqa_1.3b_test_greedy_0_1100_8_375000.catcat.txt.truncate20.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_tqa_1.3b_test_greedy_0_1100_8_375000.catcat.txt.truncate10.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_3.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_2.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_1.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_4.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_5.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_6.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_2_375000_0_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_2_375000_2_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_8_375000_2_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_6_375000_2_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_4_375000_2_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_2_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_3_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_4_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_5_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_6_short_format.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_tqa_8.3b_test_greedy_0_1100_3_80000_3_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_tqa_8.3b_test_greedy_0_1100_3_60000_3_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_tqa_8.3b_test_greedy_0_1100_3_50000_3_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_tqa_8.3b_test_greedy_0_1100_3_50000_3_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_tqa_8.3b_test_greedy_0_1100_3_100000_3_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)
    #
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_tqa_8.3b_test_greedy_0_1100_3_100000_3_short_format.txt.concat.txt.period.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications//nq_retro_ft_same_format_ctx1_357m_32_1e-5_0.0_8/generate_357m_test_greedy_0_4000_8_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx1_357m_32_1e-5_0.0/generate_357m_test_greedy_0_4000_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_test_greedy_0_4000_8_15000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx1_1.3b_32_3e-6_0.0/generate_1.3b_test_greedy_0_4000_9000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_test_greedy_0_4000_8_10500.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx0_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_105000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx0_1.3b_32_3e-6_0.0_8/generate_1.3b_test_greedy_0_4000_8_10500.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_8.3b_32_1e-6_0.0/generate_8.3b_dev_greedy_0_3000_{}.txt"
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/dev.json"

    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_add_space_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format__ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_{}.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_reuse_top_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_chunk0_ctx0_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_bert_retriever_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_4000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx2_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_6000_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx4_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx2_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_4000_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_6000_{}.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx8_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_4000_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_12/generate_8.3b_test_greedy_0_4000_12_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx{}_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_10500.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx{}_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_4000_10500.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_bert_retriever_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_4000_10500.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_bert_retriever_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_4000_8_10500.txt"
    # for step in [1, 2, 4, 8]:
    #     p_file = prediction_file.format(step)
    #     print(p_file)
    #     evaluate_ems(p_file, ground_truth_file)

    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA/dev.json"
    # for step in range(1500, 15000, 1500):
    for step in range(12000, 180000, 12000):
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_dev_greedy_0_3000_8_{}.txt"
        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_dev_greedy_0_3000_{}.txt"
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_dev_greedy_0_3000_8_{}.txt"
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx1_1.3b_32_3e-6_0.0/generate_1.3b_dev_greedy_0_3000_{}.txt"
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_dev_greedy_0_3000_8_{}.txt"
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx0_1.3b_32_3e-6_0.0/generate_1.3b_dev_greedy_0_3000_{}.txt"
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_dev_greedy_0_3000_{}.txt"
        p_file = prediction_file.format(step)
        # evaluate_ems(p_file, ground_truth_file)


    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA/test.json"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_12000_9000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_12000_8_12000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_bert_retriever_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_12000_8_9000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_bert_retriever_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_12000_12000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_test_greedy_0_12000_8_144000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_ctx1_357m_32_1e-5_0.0_8/generate_357m_test_greedy_0_12000_8_144000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx1_357m_32_1e-5_0.0/generate_357m_test_greedy_0_12000_132000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx1_1.3b_32_3e-6_0.0/generate_1.3b_test_greedy_0_12000_132000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_12000_8_144000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx1_8.3b_32_1e-6_0.0//generate_8.3b_test_greedy_0_12000_132000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_12000_8_96000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx0_1.3b_32_3e-6_0.0/generate_1.3b_test_greedy_0_12000_96000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx0_357m_32_1e-5_0.0/generate_357m_test_greedy_0_12000_96000.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/tqa_ft_same_format_ctx0_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_12000_96000.txt"
    # evaluate_ems(prediction_file, ground_truth_file)

    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx0_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_10500.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_10500.txt"


    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/dev.json"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_dev_greedy_0_3000_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_dev_greedy_0_3000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_bert_retriever_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_dev_greedy_0_3000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx2_8.3b_32_1e-6_0.0/generate_8.3b_dev_greedy_0_4000_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx0_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_6000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx8_8.3b_32_1e-6_0.0_8/generate_8.3b_dev_greedy_0_3000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_1.3b_32_3e-6_0.0_8/generate_1.3b_dev_greedy_0_3000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx1_1.3b_32_3e-6_0.0/generate_1.3b_dev_greedy_0_3000_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx1_357m_32_1e-5_0.0/generate_357m_dev_greedy_0_3000_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_retro_ft_same_format_ctx1_357m_32_1e-5_0.0_8/generate_357m_dev_greedy_0_3000_8_{}.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/nq_ft_same_format_ctx0_1.3b_32_3e-6_0.0/generate_1.3b_dev_greedy_0_3000_{}.txt"
    # for step in range(1500, 16500, 1500):
    #     p_file = prediction_file.format(step)
    #     print(p_file)
    #     try:
    #         evaluate_ems(p_file, ground_truth_file)
    #     except:
    #         continue
