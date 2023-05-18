import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--datapath", type=str, default=None, help="datapath for test json file")
    parser.add_argument("--gen_file", type=str, default=None, help="generations for test file")
    parser.add_argument("--output_file", type=str, default=None, help="output for human eval")

    args = parser.parse_args()

    return args

def save_to_output(examples, output_file):

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2)

def read_json_data(data_path):

    with open(data_path, "r") as f:
        examples = json.load(f)
    return examples 

def load_prediction(test_file):

    predictions = []
    with open(test_file, "r") as f:
        for line in f.readlines():
            predictions.append(line.strip())
    return predictions

def reformulate_example(examples, generations):

    print(len(examples), len(generations))
    assert len(examples) == len(generations)
    for example, generation in zip(examples, generations):
        del example["ctxs"]
        example["generated_answer"] = generation.strip()
    return examples

def main(args):

    test_examples = read_json_data(args.datapath)
    generations = load_prediction(args.gen_file)
    test_examples = reformulate_example(test_examples, generations)
    save_to_output(test_examples, args.output_file)

if __name__ == "__main__":

    """
    python3 combine_test_with_gen_for_eval.py  --datapath /lustre/fsw/portfolios/adlr/users/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json \
    --gen_file /lustre/fs1/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_gpt_same_format_ctx5_43b_8_1e-6/generate_43b_test_greedy_0_200_120.txt \
    --output_file ./att_test_eval.json
    """

    args = get_args()
    main(args)