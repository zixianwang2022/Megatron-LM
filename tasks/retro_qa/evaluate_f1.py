from tqdm import tqdm
import string
import json
from metrics import F1Metric

def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))

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
        data.append(instance["answers"][0])

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

    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/benz/test.json"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/bf16_nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_benz.txt"
    prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/prefix_tuning/megatron-lm/tasks/prompt_learning/checkpoints/benz_retrieved_adapter_530b_32_3e-5_48/generate_530b_test_greedy_0_400.txt"
    # evaluate_f1(ground_truth_file, prediction_file, True)
    # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_benz_test_204.txt"
    # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_nq/generate_benz_test_204.txt"
    # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_benz_test_612.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_test_272.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx0_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_test_340.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_ft_same_format_ctx0_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_136.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_68.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_68.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_qie_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_test_459.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_qie_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_benz_test_459.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_longform_nq/generate_benz_test_765.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_benz_test_204.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_8.3b_test_greedy_0_250_1_136.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_msmarcominilm_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_8.3b_test_greedy_0_250_1_68.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_msmarcominilm_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_68.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_test_greedy_0_250_8_340.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_test_greedy_0_250_8_340.txt"
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/qa_retro/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_test_greedy_0_4000_8_340.txt"
    print(prediction_file)
    # evaluate_f1(ground_truth_file, prediction_file)


    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/benz/dev.json"
    for ckpt_step in range(1, 6):
        ckpt_step = 68 * ckpt_step
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_nq/generate_benz_dev_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx0_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_msmarcominilm_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_dev_greedy_0_250_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_dev_greedy_0_250_8_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_8.3b_dev_greedy_0_250_1_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)

        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_ft_same_format_ctx0_8.3b_8_1e-6_0.0/generate_8.3b_dev_greedy_0_250_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_dev_greedy_0_250_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_retro_ft_same_format_ctx0_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        # ckpt_step = 153 * ckpt_step
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_qie_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_qie_ft_same_format_ctx0_8.3b_8_1e-6_0.0_1_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_qie_ft_same_format_ctx0_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_qie_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_longform_nq/generate_benz_dev_{}.txt".format(ckpt_step)
        # prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_benz_dev_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_msmarcominilm_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_8.3b_dev_greedy_0_250_1_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_msmarcominilm_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_dev_greedy_0_250_8_{}.txt".format(ckpt_step)
        prediction_file ="/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_dpr_finetuned_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_dev_greedy_0_250_{}.txt".format(ckpt_step)
        # print(prediction_file)
        # groundtruth_answer = load_groundtruth_file(ground_truth_file)
        # predicted_answers = load_prediction(prediction_file)
        # try:
        #     compute_f1_score(predicted_answers, groundtruth_answer)
        # except:
        #     continue

    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover/dev.json"
    for ckpt_step in range(1, 6):
        # ckpt_step = 79 * ckpt_step
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_msmarcominilm_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_dev_greedy_0_250_8_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_msmarcominilm_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_8.3b_dev_greedy_0_250_1_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_dev_greedy_0_250_{}.txt".format(ckpt_step)
        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_msmarcominilm_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_dev_greedy_0_250_{}.txt".format(ckpt_step)
        # print(prediction_file)
        # groundtruth_answer = load_groundtruth_file(ground_truth_file)
        # predicted_answers = load_prediction(prediction_file)
        # try:
        #     compute_f1_score(predicted_answers, groundtruth_answer)
        # except:
        #     continue

    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover/test.json"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_316.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_msmarcominilm_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_395.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_msmarcominilm_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_8.3b_test_greedy_0_250_1_237.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_msmarcominilm_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_test_greedy_0_250_8_237.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_plus_benz_tasb_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_292.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/retro/checkpoints/applications/landrover_plus_benz_tasb_ftmsmarcominilm_chunkbysents150_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_test_greedy_0_250_8_584.txt"
    prediction_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/retro/checkpoints/applications/benz_clean_plus_landrover_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/landrover_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_generate_8.3b_test_greedy_0_250_8_513.txt"
    prediction_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/benz_clean_plus_landrover_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_gpt_same_format_ctx1_8b_16_3e-6/generate_8b_test_greedy_0_250_85.txt"
    print(prediction_file)
    ground_truth_file = "//lustre/fs1/portfolios/adlr/users/pengx/projects/retro/data/benz_clean_plus_landrover_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    # evaluate_f1(ground_truth_file, prediction_file)

    for ckpt_step in range(1, 6):
        # ckpt_step = 68 * ckpt_step
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_test_greedy_0_250_8_{}.txt".format(ckpt_step)
        # ckpt_step = 79 * ckpt_step
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_tasb_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla//generate_8.3b_test_greedy_0_250_1_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_384//generate_8.3b_test_greedy_0_250_1_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_4_384//generate_8.3b_test_greedy_0_250_4_{}.txt".format(ckpt_step)
        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_1_vanilla/generate_8.3b_test_greedy_0_250_1_{}.txt".format(ckpt_step)

        # ckpt_step = 146 * ckpt_step
        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_plus_benz_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_4_384/generate_8.3b_test_greedy_0_250_4_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_plus_benz_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/generate_8.3b_test_greedy_0_250_8_{}.txt".format(ckpt_step)
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_plus_benz_tasb_retrieved_retro_ft_same_format_ctx1_1.3b_8_3e-6_0.0_4_384/generate_1.3b_test_greedy_0_250_4_{}.txt".format(ckpt_step)
        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_plus_benz_tasb_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_{}.txt".format(ckpt_step)


        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_tasb_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_{}.txt".format(ckpt_step)
        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_tasb_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_{}.txt".format(ckpt_step)
        # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/benz/test.json"
        # ckpt_step = 171 * ckpt_step
        task_name = "clean_benz_ford_landrover"
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/{}_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/clean_benz_tasb_retrieved_generate_8.3b_test_greedy_0_250_8_{}.txt".format(task_name, ckpt_step)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/clean_benz_tasb_retrieved/test.json"

        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/{}_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/ford_tasb_retrieved_generate_8.3b_test_greedy_0_250_8_{}.txt".format(task_name, ckpt_step)
        # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/ford_tasb_retrieved/test.json"

        # prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/{}_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/landrover_tasb_retrieved_generate_8.3b_test_greedy_0_250_8_{}.txt".format(task_name, ckpt_step)
        # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover_tasb_retrieved/test.json"

        # ckpt_step = 146 * ckpt_step
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/benz_plus_landrover_tasb_ftmsmarcominilm_chunkbysents150_retrieved_prefix_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_8_vanilla/landrover_tasb_retrieved_generate_8.3b_test_greedy_0_250_8_{}.txt".format(ckpt_step)

        checkpoint_step = 39 * ckpt_step
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/43b_gpt_QA/checkpoints/applications/landrover_tasb_retrieved_gpt_same_format_ctx1_8b_16_3e-6/generate_8b_test_greedy_0_250_{}.txt".format(checkpoint_step)

        checkpoint_step = 79 * ckpt_step
        prediction_file = "/lustre/fsw/adlr/adlr-nlp/pengx/43b_gpt_QA/checkpoints/applications/landrover_tasb_retrieved_gpt_same_format_ctx1_8b_8_3e-6/generate_8b_test_greedy_0_250_{}.txt".format(checkpoint_step)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover/test.json"

        checkpoint_step = 85 * ckpt_step
        prediction_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/benz_clean_plus_landrover_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_gpt_same_format_ctx1_8b_16_3e-6/generate_8b_test_greedy_0_250_{}.txt".format(checkpoint_step)    
        checkpoint_step = 69 * ckpt_step
        # prediction_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/benz_clean_plus_landrover_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_gpt_same_format_ctx1_8b_16_3e-6/generate_8b_test_greedy_0_250_{}.txt".format(checkpoint_step)

        checkpoint_step = 170 * ckpt_step
        prediction_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_gpt_same_format_ctx10_43b_8_1e-6/generate_43b_test_greedy_0_250_{}.txt".format(checkpoint_step)
        ground_truth_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/retro/data/landrover_tasb_retrieved/test.json"

        checkpoint_step = 170 * ckpt_step
        # prediction_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_gpt_same_format_ctx1_43b_8_1e-6/generate_43b_test_greedy_0_250_{}.txt".format(checkpoint_step)
        # ground_truth_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/retro/data/landrover_tasb_retrieved/test.json"

        # checkpoint_step = 170 * ckpt_step
        # prediction_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_gpt_same_format_ctx10_43b_8_1e-6/benz_clean_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_generate_43b_test_greedy_0_250_{}.txt".format(checkpoint_step)
        # ground_truth_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/retro/data/benz_clean_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"

        # prediction_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_gpt_same_format_ctx15_43b_8_1e-6/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_generate_43b_test_greedy_0_250_{}.txt".format(checkpoint_step)
        # ground_truth_file = "/lustre/fs1/portfolios/adlr/users/pengx/projects/retro/data/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"

        print(prediction_file)
        print(ground_truth_file)
        try:
            evaluate_f1(ground_truth_file, prediction_file)
        except:
            continue

