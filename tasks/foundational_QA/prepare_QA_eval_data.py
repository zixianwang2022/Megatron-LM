from datasets import load_dataset
import os, json


def load_mrqa(split="test"):

    mrqa = load_dataset("mrqa", split=split)
    return mrqa

def get_subset(mrqa, subset_key):

    valid_items = []
    for item in mrqa:
        if item["subset"] == subset_key:
            valid_items.append(item)
    return valid_items
    

def formulate_fqa_item(question, answer, context, ctxs=None):

    fqa_data_format = {"sub-paragraphs": "",
        "question": "",
        "answers": [""],
        "ctxs": [
        {
            "title": "",
            "text": ""
        }]
    }
    fqa_data_format["question"] = question
    fqa_data_format["answers"] = [answer]
    fqa_data_format["sub-paragraphs"] = context
    if ctxs is None:
        fqa_data_format["ctxs"][0]["text"] = context
    else:
        def get_ctx_example(ctx_sample):
            return {
                "title": "",
                "text": ctx_sample
            } 
        fqa_data_format["ctxs"] = [get_ctx_example(ctx_sample) for ctx_sample in ctxs]
    return fqa_data_format


def items_to_fqa(items):

    all_fqa_items = []
    for item in items:
        context = item["context"]
        question = item["question"]
        answer = item["answers"][0]
        ctxs = None
        if "ctxs" in item:
            ctxs = item["ctxs"]
        fqa_item = formulate_fqa_item(question, answer, context, ctxs)
        all_fqa_items.append(fqa_item)

    return all_fqa_items

def load_boolq(split="validation"):

    boolq = load_dataset("boolq", split=split)
    boolq_items = []
    for row in boolq:
        item = {}
        item["context"] = row["passage"]
        item["answers"] = [str(row["answer"]).lower()]
        item["question"] = row["question"]
        boolq_items.append(item)

    return boolq_items

def load_race(split="test"):

    race = load_dataset("race","all", split=split)
    race_items = []

    def format_multichoice(multichoice_options):
        options_text = ["({}) {}".format(chr(ord('A')+i), option) for i, option in zip(range(len(multichoice_options)), multichoice_options)]
        return "Choose one based on the following options: {}".format(" ".join(options_text))
 
    for row in race:
        item = {}
        item["context"] = row["article"]
        item["answers"] = ["({})".format(str(row["answer"])) + " " + row["options"][ord(row["answer"]) - ord("A")]]
        item["question"] = row["question"] + " " + format_multichoice(row["options"])
        race_items.append(item)

    return race_items

def load_msmarco(split="validation"):

    msmarco = load_dataset("ms_marco", 'v1.1', split=split)
    msmarco_items = []
    for row in msmarco:
        item = {}
        item["answers"] = row["answers"]
        if len(item["answers"]) < 1:
            continue
        item["question"] = row["query"]
        item["ctxs"] = row["passages"]["passage_text"]
        item["context"] = "\t".join([ctx for ctx, is_selected in zip(item["ctxs"], row["passages"]["is_selected"]) if is_selected])
        msmarco_items.append(item)

    return msmarco_items

def load_multirc(split="validation"):
    ## one relevant https://github.com/google-research/FLAN/blob/main/flan/templates.py
    multirc = load_dataset("super_glue", "multirc", split=split)
    multirc_items = []
    textual_answers = ["false", "true"]
    for row in multirc:
        item = {}
        item["question"] = "Given the question: '{}' and the response: '{}'. Is the response to the question factually correct?".format(row["question"], row["answer"])
        item["context"] = row["paragraph"]
        item["answers"] = [textual_answers[row["label"]]]
        multirc_items.append(item)

    return multirc_items

def main(task="mrqa", subset_key="TextbookQA", base_dir="./eval_data/", split="test"):

    if task == "mrqa":
        if split == "train":
            mrqa = load_mrqa(split=split)
        else:
            mrqa = load_mrqa()
        # {'DuoRC.ParaphraseRC', 'TextbookQA', 'BioASQ', 'RelationExtraction', 'RACE', 'DROP'}
        valid_items = get_subset(mrqa, subset_key)
        
    elif task == "boolq":
        if split == "train":
            valid_items = load_boolq(split=split)
        else:
            valid_items = load_boolq()
        subset_key = "boolq"

    elif task == "msmarco":
        if split == "train":
            valid_items = load_msmarco(split=split)
        else:
            valid_items = load_msmarco()
        subset_key = task

    elif task == "race":
        if split == "train":
            valid_items = load_race(split=split)
        else:
            valid_items = load_race()
        subset_key = task

    elif task == "multirc":
        if split == "train":
            valid_items = load_multirc(split=split)
        else:
            valid_items = load_multirc()
        subset_key = task

    else:
        raise ValueError("invalid task")

    all_fqa_items = items_to_fqa(valid_items)
    # print(valid_items[0])
    # print(all_fqa_items[0])

    if subset_key == "DuoRC.ParaphraseRC":
        subset_key = "DuoRC_ParaphraseRC"
    output_dir = base_dir + subset_key
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if split == "train":
        output_file = "{}/train.json".format(output_dir)
    else:
        output_file = "{}/test.json".format(output_dir)
    print(output_file)
    with open(output_file, "w") as f:
        json.dump(all_fqa_items, f, indent=2)


def test():

    mrqa = load_mrqa()
    # {'DuoRC.ParaphraseRC', 'TextbookQA', 'BioASQ', 'RelationExtraction', 'RACE', 'DROP'}
    subset_key = 'DuoRC.ParaphraseRC'
    valid_items = get_subset(mrqa, subset_key)
    print(valid_items[0])
    all_fqa_items = items_to_fqa(valid_items)
    print(all_fqa_items[0])

if __name__ == "__main__":

    # main("mrqa", subset_key="DuoRC.ParaphraseRC")
    # test()
    # main("boolq")
    # main("msmarco")
    # main("race")
    # main("multirc")
    split = "train"
    # main("boolq", split=split)
    # main("msmarco", split=split)
    # main("race", split=split)
    # main("multirc", split=split)
    main("mrqa", subset_key="DuoRC.ParaphraseRC", split=split)
    main("mrqa", subset_key="TextbookQA", split=split)
    main("mrqa", subset_key="BioASQ", split=split)