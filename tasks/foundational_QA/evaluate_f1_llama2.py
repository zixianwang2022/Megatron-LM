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
    return f1


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
        try:
            return compute_f1_score(predicted_answers, groundtruth_answer)
        except:
            return 0.0
    # groundtruth_answer, predicted_answers = groundtruth_answer[:43], predicted_answers[:43]
    # compute_f1_score(predicted_answers, groundtruth_answer)
        
if __name__ == "__main__":

        import argparse
        argparser = argparse.ArgumentParser(description="run fqa eval")
        argparser.add_argument("--step", type=int, default=1000)
        args = argparser.parse_args()
        all_f1 = []
        model_name = "qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6"
        model_name = "long_qc_v3_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_step_1000"
        model_name = "multiturn_qa_blend_commercial_v5_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_128_5e-6_step_500"
        model_name = "multiturn_qa_blend_v2_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_128_5e-6_step_500"
        model_name = "multiturn_qa_blend_v2_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_64_3e-7_step_500"
        model_name = "multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
        model_name = "primitive_stingray16k_fqa_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_64_3e-7_step_500"
        # model_name = "long_fqa_research_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_64_3e-7_step_500"
        ckpt_path="/lustre/fsw/portfolios/adlr/users/pengx/projects/sft_70b_qa/checkpoints/applications/{}/".format(model_name)
        n_ctx=5
        step=args.step
        prediction_file = ckpt_path + "/doc2dial_{}_generate_70b_test_greedy_0_1000_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/doc2dial/retrieval/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/quac_{}_generate_70b_test_greedy_0_1000_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/quac/quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/qrecc_{}_generate_70b_test_greedy_0_1000_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/sharc_{}_generate_70b_test_greedy_0_1000_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/sharc/sharc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_70b_test_greedy_0_250_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_70b_test_greedy_0_250_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_70b_test_greedy_0_250_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/nv_benefits_dragon_retriever300_retrieved_generic/test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_70b_test_greedy_0_250_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_70b_test_greedy_0_250_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_70b_test_greedy_0_250_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        prediction_file = ckpt_path + "/nq_{}_generate_70b_test_greedy_0_200_{}_ret.txt_0920".format(n_ctx, step)
        ground_truth_file = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/foundational_qa/test_benchmarks/nq/test.json"
        print(prediction_file)
        print(ground_truth_file)
        all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        # prediction_file = ckpt_path + "/hotpotqa_{}_generate_70b_test_greedy_0_200_{}_ret.txt_0920".format(n_ctx, step)
        # ground_truth_file = "/lustre/fsw/portfolios/adlr/users/pengx/data/scroll_eval_data/hotpotqa.dragon_retriever_chunkbysents300/test.json"
        # print(prediction_file)
        # print(ground_truth_file)
        # all_f1.append(evaluate_f1(ground_truth_file, prediction_file))

        print("average f1", sum(all_f1) / len(all_f1))
