"""
Using huggingface evaluate metrics

First make sure you install the huggingface evaluate package:
    pip install evaluate

Then install the corresponding packages for different metrics.
For example:
    pip install rouge_score
    pip install bert_score
"""

import numpy as np
import evaluate
from metrics import F1Metric
import json
import argparse
import os
from numpy import mean

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--datapath", type=str, default=None, help="datapath for test json file")
    parser.add_argument("--gen_test_file", type=str, default=None, help="generations for test file")

    args = parser.parse_args()

    return args

def evaluate_nlg(metric, predictions, references, model_type="distilbert-base-uncased"):
    """
        model_type: for bertscore evaluation
    """
    eval_metric = evaluate.load(metric)

    if metric == "bertscore":
        results = eval_metric.compute(predictions=predictions, references=references, model_type=model_type)
    elif metric == "rouge":
        results = eval_metric.compute(predictions=predictions, references=references, use_aggregator=False)
        for key in results:
            results[key] = mean(results[key])
    else:
        results = eval_metric.compute(predictions=predictions, references=references)

    return results

def read_json_data(data_path):
    references = []
    questions = []
    with open(data_path, "r") as f:
        examples = json.load(f)
        for data_item in examples:
            questions.append(data_item['question'])
            if "answers" in data_item:
                references.append(data_item['answers'][0])
            elif "answer" in data_item:
                references.append(data_item['answer'])
            else:
                raise ValueError("need answer or answers from input json")

    return questions, references    

def load_prediction(test_file):

    predictions = []
    with open(test_file, "r") as f:
        for line in f.readlines():
            predictions.append(line.strip())
    return predictions

def evaluate_all_metrics(predictions, references):
    #print("test:", len(predictions), len(references))
    precision, recall, f1 = F1Metric.compute_all_pairs(predictions, references)
    rouge_results = evaluate_nlg("rouge", predictions, references)
    bertscore_results = evaluate_nlg("bertscore", predictions, references)
    bertscore_results_avg = {'precision': np.mean(bertscore_results['precision']),
                             'recall': np.mean(bertscore_results['recall']),
                             'f1': np.mean(bertscore_results['f1'])}
    return precision, recall, f1, rouge_results, bertscore_results, bertscore_results_avg


if __name__ == "__main__":

    """
    command:
    python3 evaluate_att.py --datapath /lustre/fsw/portfolios/adlr/users/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json \
    --gen_test_file /lustre/fsw/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_gpt_same_format_ctx1_43b_8_1e-6/generate_43b_test_greedy_0_200_120.txt
    """
    args = get_args()
    _, references = read_json_data(args.datapath)
    predictions = load_prediction(args.gen_test_file)
    precision, recall, f1, rouge_results, bertscore_results, bertscore_results_avg = evaluate_all_metrics(predictions, references)

    saved_metrics_names = ["filename", "f1", "rouge1", "rouge2", "rougeL", "rougeLsum", "bert_score_f1"]
    print(",".join(saved_metrics_names))
    metrics_scores = [f1, rouge_results["rouge1"], rouge_results["rouge2"], rouge_results["rougeL"], rouge_results["rougeLsum"], bertscore_results_avg["f1"]]
    metrics_scores = [args.gen_test_file] + ["{:.3}".format(float(item)) for item in metrics_scores]
    print(",".join(metrics_scores))
    
