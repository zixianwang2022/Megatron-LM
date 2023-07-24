from tqdm import tqdm
import string
import json
from metrics import F1Metric

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
    if not reduced_test_only:
        compute_f1_score(predicted_answers, groundtruth_answer)
    groundtruth_answer, predicted_answers = groundtruth_answer[:43], predicted_answers[:43]
    compute_f1_score(predicted_answers, groundtruth_answer)
        
if __name__ == "__main__":

        model_name = "qa_blendv0_gpt_1e-8_conv_quiet_cockatoo_pp1_same_format_ctx1_43b_64_3e-7"
        model_name = "qa_blendv12_gpt_1e-8_conv_quiet_cockatoo_pp1_fixed_newsqa_same_format_ctx1_43b_64_3e-7"
        model_name = "qa_blendv13_gpt_1e-8_conv_quiet_cockatoo_pp1_same_format_ctx1_43b_64_3e-7"
        model_name = "qa_blendv12_gpt_1e-8_conv_quiet_cockatoo_pp1_fixed_doc2dial_same_format_ctx1_43b_64_3e-7"
        ckpt_path="/lustre/fsw/adlr/adlr-nlp/pengx/sft_43b_qa/checkpoints/applications/{}/".format(model_name)
        n_ctx=5
        prediction_file = ckpt_path + "/inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_generate_43b_test_greedy_0_250_4500_ret.txt.v2".format(n_ctx)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/inference_input_retriever_dragon_msmarcominilm_doc2dial/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_4500_ret.txt.v2".format(n_ctx)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_4500_ret.txt.v2".format(n_ctx)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_43b_test_greedy_0_250_4500_ret.txt.v2".format(n_ctx)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/nv_benefits_dragon_retriever300_retrieved_generic/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_4500_ret.txt.v2".format(n_ctx)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_4500_ret.txt.v2".format(n_ctx)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_4500_ret.txt.v2".format(n_ctx)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        prediction_file = ckpt_path + "/nq_{}_generate_43b_test_greedy_0_200_4500_ret.txt.v2".format(n_ctx)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)
