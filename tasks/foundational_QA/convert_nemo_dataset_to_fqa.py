import json


def read_nemo_data(input_filename):

    data = []
    with open(input_filename, "r") as f:
        for line in f.readlines():
            item = json.loads(line)
            data.append(item)

    return data


def to_fqa_data(input_text, output_text):

    item = {"paragraph_id": "", "question": input_text, "answer": output_text, "sub-paragraphs": "", "word count": "", "Date": ""}

    return item


def convert_nemo_to_fqa(data):

    fqa_data = []
    for item in data:
        item = to_fqa_data(item["input"], item["output"])
        fqa_data.append(item)

    return fqa_data


def write_fqa_data(fqa_data, output_filename):

    with open(output_filename, "w") as f:
        json.dump(fqa_data,f, indent=2)

def split_train_dev(fqa_data):

    total = len(fqa_data)
    num_train = int(0.8 * total)
    train_data = fqa_data[:num_train]
    dev_data = fqa_data[num_train:]

    return train_data, dev_data

if __name__ == "__main__":

    input_filename = "/lustre/fsw/swdl/swdl-langspeech/datasets/data/BigNLP/tool_generated_sft_datasets/quiet-cockatoo/quiet-cockatoo_commercial.shuf.jsonl"
    data = read_nemo_data(input_filename)
    fqa_data = convert_nemo_to_fqa(data)
    train_data, dev_data = split_train_dev(fqa_data)
    output_filename = "/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/quiet_cockatoo/quiet_cockatoo_QA_{}.json"
    write_fqa_data(train_data, output_filename.format("train"))
    write_fqa_data(dev_data, output_filename.format("dev"))
