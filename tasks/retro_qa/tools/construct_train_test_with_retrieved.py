from tqdm import tqdm
import string
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--full_retrieved_data_file", type=str, default=None, help="full generation")
    parser.add_argument("--output_file", type=str, default=None, help="full generation")
    parser.add_argument("--index_file", type=str, default=None, help="index for the output file")

    args = parser.parse_args()

    return args

def load_index_file(test_file):

    rows = []
    with open(test_file, "r") as f:
        rows = json.load(f)
        idxs = [item["id"] for item in rows]
    return idxs

def load_retrieved_data_file(retrieved_data_file):

    with open(retrieved_data_file, "r") as f:
        rows = json.load(f)
    return rows

def filter_with_idx(examples, idxs):

    examples_filtered = [examples[i] for i in idxs]
    return examples_filtered

def save_exampled(examples, output_file):

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2)

def rewrite_keys(examples, key="answer"):

    for example in examples:
        example["answers"] = [example["answer"]]
        del example["answer"]
    return examples

def main(args):

    full_examples = load_retrieved_data_file(args.full_retrieved_data_file)
    idxs = load_index_file(args.index_file)
    examples_filtered = filter_with_idx(full_examples, idxs)
    examples_filtered = rewrite_keys(examples_filtered)
    save_exampled(examples_filtered, args.output_file)

if __name__ == "__main__":

    args = get_args()
    main(args)

    """
    retriever=dragon_retriever_msmarcominilm_reranker
    chunk_size=150
    split=train
    full_retrieved_data_file=/lustre/fsw/portfolios/adlr/users/pengx/data/att/experiments/att_qa_${retriever}_chunkbysents${chunk_size}_retriever_top10.json
    index_file=/lustre/fsw/portfolios/adlr/users/pengx/data/att/raw/att_qa_${split}.json
    output_file=/lustre/fsw/portfolios/adlr/users/pengx/data/att/att_${retriever}_chunkbysents${chunk_size}_retrieved/${split}.json
    mkdir -p /lustre/fsw/portfolios/adlr/users/pengx/data/att/att_${retriever}_chunkbysents${chunk_size}_retrieved/
    python construct_train_test_with_retrieved.py --full_retrieved_data_file ${full_retrieved_data_file}\
    --index_file $index_file --output_file $output_file
    """