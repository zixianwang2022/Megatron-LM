import json

def load_data(data_file):

    with open(data_file, "r") as f:
        examples = json.load(f)

    return examples

import random

def dump_data(data, data_file, shuffle=True):
    
    if shuffle:
        random.shuffle(data)
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":

    prefix = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/"
    # data_files = ["data/benz/train.json", "data/landrover/train.json", "data/landrover/dev.json", "data/landrover/test.json"]
    # data_files = ["data/benz_tasb_retrieved/train.json", "data/landrover_tasb_retrieved/train.json"]
    # data_files = ["data/clean_benz_tasb_retrieved/train.json", "data/landrover_tasb_retrieved/train.json"]
    # output_file = "data/clean_benz_landrover_tasb_retrieved/train.json"
    data_files = ["data/clean_benz_tasb_retrieved/train.json", "data/landrover_tasb_retrieved/train.json", "data/ford_tasb_retrieved/train.json"]
    output_file = "data/clean_benz_ford_landrover_tasb_retrieved/train.json"
    all_examples = []
    for data_file in data_files:
        print(data_file)
        examples = load_data(prefix + data_file)
        all_examples.extend(examples)

    dump_data(all_examples, prefix + output_file)
