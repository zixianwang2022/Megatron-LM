
import os
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import sacrebleu
from nltk.tokenize import word_tokenize
import evaluate

## pip install pycocoevalcap
## pip install evaluate

def evaluate_scores(pred_list, gold_list):
    
    gold_dict_file = './tmp/gold.json'
    gold_dict = {"annotations": [], "images": []}
    for id_, gold in enumerate(gold_list):
        gold_dict["annotations"].append({"image_id": id_, "caption": gold, "id": id_})
        gold_dict["images"].append({"id": id_})
    with open(gold_dict_file, "w") as f:
        f.write(json.dumps(gold_dict))

    pred_list_file = './tmp/pred.json'
    pred_list_tmp = []
    for id_, pred in enumerate(pred_list):
        data_item = {"image_id": id_, "caption": pred}
        pred_list_tmp.append(data_item)
    with open(pred_list_file, "w") as f:
        f.write(json.dumps(pred_list_tmp))
        
    # create coco object and coco_result object
    coco = COCO(gold_dict_file)
    coco_result = coco.loadRes(pred_list_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')


def evaluate_bleu(pred_list, gold_list):
    
    bleu = evaluate.load("bleu")
    pred_final_list = []
    for pred in pred_list:
        pred = " ".join(word_tokenize(pred.lower()))
        pred_final_list.append(pred)

    print("prediction 1st sentence:", pred_final_list[0])   

    gold_final_list = []
    for gold in gold_list:
        gold = " ".join(word_tokenize(gold.lower()))
        gold_final_list.append([gold])

    print("Reference 1st sentence:", gold_final_list[0])

    results = bleu.compute(predictions=pred_final_list, references=gold_final_list)
    print(results)


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


def evaluate_nlg(ground_truth_file, prediction_file):

    groundtruth_answers = load_groundtruth_file(ground_truth_file)
    predicted_answers = load_prediction(prediction_file)

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]    

    evaluate_scores(predicted_answers, groundtruth_answers)
    # evaluate_bleu(predicted_answers, groundtruth_answers)


if __name__ == "__main__":

    # model_name = "multiturn_qa_blendv2_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    # model_name = "multiturn_qa_blendv1_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7_old"
    # model_name = "multiturn_qa_blendv1_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    model_name = "multiturn_qa_blend_commercial_v15_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"

    ckpt_path="/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/checkpoints/applications/{}".format(model_name)
    n_ctx=5
    
    ## doc2dial w/ retrieval
    # prediction_file = ckpt_path + "/doc2dial_{}_generate_8b_test_greedy_0_1000_3000_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_nlg(ground_truth_file, prediction_file)
    
    ## doc2dial w/ gold context
    prediction_file = ckpt_path + "/doc2dial_gold_{}_generate_8b_test_greedy_0_4000_4000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_goldctx_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_nlg(ground_truth_file, prediction_file)

    # ## doc2dial w/ gold context v2
    # prediction_file = ckpt_path + "/doc2dial_gold_v2_{}_generate_8b_test_greedy_0_4000_4000_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_goldctx_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_nlg(ground_truth_file, prediction_file)

    