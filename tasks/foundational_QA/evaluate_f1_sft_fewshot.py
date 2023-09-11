
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tqdm import tqdm
import string
import json
from msdp.metrics import F1Metric
from evaluate_f1_fqa_zeroshot import evaluate_cannot_answer_acc

def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))
    if len(predicted_answers) != len(groundtruth_answer):
        groundtruth_answer = groundtruth_answer[:len(predicted_answers)]

    guess_list = []
    for answer in predicted_answers:
        answer = answer.strip()
        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")
        guess_list.append(answer)

    answer_list = []
    for answer in groundtruth_answer:
        answer = answer.strip()
        if answer == "no_passages_used":
            answer = ""
        answer_list.append(answer)

    assert len(guess_list) == len(answer_list), \
        "lengths of guess and answer are different!"

    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list)
    print('Method: %s; Precision: %.4f; recall: %.4f; f1: %.4f' % (\
        exp_name, precision, recall, f1))


def load_groundtruth_file(data_file):
    
    with open(data_file, "r") as f:
        nq_examples = json.load(f)

    data = []
    for instance in nq_examples:
        if "answers" in instance:
            answers = instance["answers"]
        elif "answer" in instance:
            if type(instance["answer"]) is str:
                answers = [instance["answer"]]
            elif type(instance["answer"]) is list:
                answers = instance["answer"]
            else:
                answers = [str(instance["answer"])]
        else:
            raise ValueError("need to have answer or answers")
        data.append(answers[0])

    return data

def load_prediction(data_file):

    data = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            data.append(line.strip())

    return data

def evaluate_f1(ground_truth_file, prediction_file, reduced_test_only=False):

    groundtruth_answer = load_groundtruth_file(ground_truth_file)
    predicted_answers = load_prediction(prediction_file)
    # if not reduced_test_only:
    #     compute_f1_score(predicted_answers, groundtruth_answer)
    # groundtruth_answer, predicted_answers = groundtruth_answer[:43], predicted_answers[:43]
    compute_f1_score(predicted_answers, groundtruth_answer)

if __name__ == "__main__":

    model_name = "quiet_cockatoo_pp1"
    ckpt_path="/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/checkpoints/applications/{}".format(model_name)
    n_ctx=5
    
    ## single-turn (batch-1)
    # prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_fewshot_43b_test_greedy_0_250_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_fewshot_43b_test_greedy_0_250_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/nq_{}_generate_fewshot_43b_test_greedy_0_200_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_fewshot_43b_test_greedy_0_250_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/nv_benefits_dragon_retriever300_retrieved_generic/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_fewshot_43b_test_greedy_0_250_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_fewshot_43b_test_greedy_0_250_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_fewshot_43b_test_greedy_0_250_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)
    
    ## single-turn (batch-2)
    prediction_file = ckpt_path + "/BioASQ_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/BioASQ/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/DuoRC_ParaphraseRC_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/DuoRC_ParaphraseRC/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/boolq_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/boolq/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/msmarco_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/msmarco/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/multirc_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/multirc/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/race_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/race/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/TextbookQA_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/TextbookQA/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    ## multi-turn
    # prediction_file = ckpt_path + "/doc2dial_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/quac_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)
    # print("accuracy on cannot answer:")
    # evaluate_cannot_answer_acc(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/qrecc_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/sharc_{}_generate_fewshot_43b_test_greedy_0_1000_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sharc/sharc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)
    # print("accuracy on cannot answer:")
    # evaluate_cannot_answer_acc(ground_truth_file, prediction_file)
