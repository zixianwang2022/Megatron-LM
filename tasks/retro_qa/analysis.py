from evaluate import read_prediction_withprob, read_prediction, ems
import json

def compare_answers(prediction_file_1, prediction_file_2, ground_truth_file):

    prediction_list_1 = read_prediction(prediction_file_1)
    prediction_list_2 = read_prediction(prediction_file_2)

    ground_truths_list = []
    question_list = []
    nbr_list = []

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

        question_list.append(each["question"])
        nbr_list.append(each["ctxs"][:9])

    exactmatch = []

    good_example_list = []

    assert len(prediction_list_1) == len(prediction_list_2)
    cnt = 0
    for i, (p1, p2) in enumerate(zip(prediction_list_1, prediction_list_2)):
        s1 = ems(p1, ground_truths_list[i])
        s2 = ems(p2, ground_truths_list[i])
        if s1 and not s2:
            print('-' * 30)
            print("question:", question_list[i])
            print("ground truth:", ground_truths_list[i])
            print("retro:", p1, ',', s1)
            print("gpt:", p2, ',', s2)
            for j, evidence in enumerate(nbr_list[i][:5]):
                print("evidence", j, evidence)
            # print(nbr_list[i])
            cnt += 1
    print(cnt)
        

def nq_analysis():

    ckpt_dir = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/"
    prediction_file_1 = ckpt_dir + "/nq_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_4000_8_10500.txt"
    prediction_file_2 = ckpt_dir + "/nq_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_6000_10500.txt"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    compare_answers(prediction_file_1, prediction_file_2, ground_truth_file)

def tqa_analysis():

    prediction_file_1 = ckpt_dir + "/tqa_retro_ft_same_format_ctx1_8.3b_32_1e-6_0.0_8/generate_8.3b_test_greedy_0_12000_8_12000.txt"
    prediction_file_2 = ckpt_dir + "/tqa_ft_same_format_ctx1_8.3b_32_1e-6_0.0/generate_8.3b_test_greedy_0_12000_9000.txt"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA/test.json"
    compare_answers(prediction_file_1, prediction_file_2, ground_truth_file)


if __name__ == "__main__":

    ### NQ
    nq_analysis()

    ### TQA
    # tqa_analysis()

    pass
