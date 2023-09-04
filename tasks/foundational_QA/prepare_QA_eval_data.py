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
    

def formulate_kb_item(paragraph, paragraph_id):

    kb_format = {
        "source": "",
        "paragraph_id": paragraph_id,
        "text": paragraph,
        "title": None,
        "word_count": len(paragraph.split())
    }
    return kb_format

    

def formulate_fqa_item(question, answer, context, ctxs=None, instruction=None):

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
    if instruction is not None:
        fqa_data_format["instruction"] = instruction
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
        # for ctx, is_selected in zip(item["ctxs"], row["passages"]["is_selected"]):
        #     if is_selected:
        #         print(ctx, is_selected)
        #         print(item["ctxs"], row["passages"]["is_selected"])
        # print(len([ctx for ctx, is_selected in zip(item["ctxs"], row["passages"]["is_selected"]) if is_selected]))
        item["context"] = "\t".join([ctx for ctx, is_selected in zip(item["ctxs"], row["passages"]["is_selected"]) if is_selected])
        # if len([ctx for ctx, is_selected in zip(item["ctxs"], row["passages"]["is_selected"]) if is_selected]) > 1:
        #     print(item["context"])    
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

def load_zero_scroll(task="qasper", split="validation", kb_only=False):

    if task not in ["qasper", "narrative_qa", "musique", "quality"]:
        raise ValueError('invalid task name for zero_scroll, choose from "qasper", "narrative_qa", "musique", "quality"')

    zero_scroll_items = []
    dataset = load_dataset("tau/zero_scrolls", task)["validation"]
    for sample in dataset:
        prompted_input = sample["input"]
        items = prompted_input.split("\n\n")
        instruction = items[0]
        answer = sample["output"]

        if task == "quality":
            question = "\n\n".join(items[-3:-1])[len("Question and Possible Answers:\n"):]
            paragraph = ["\n\n".join(items[1:-3])]
        elif task == "musique":
            question = items[-2][len("Question:\n"):]
            paragraph = items[1:-2]
        elif task == "narrative_qa":
            question = items[-2][len("Question:\n"):]
            paragraph = ["\n\n".join(items[1:-2])]
        elif task == "qasper":
            question = items[-2][len("Question:\n"):]
            paragraph = [items[1]] ## items[1] is the same as items[-3]
        else:
            raise ValueError("invalid task")

        item = {}
        item["context"] = "\n\n".join(paragraph)
        item["answers"] = [answer]
        item["question"] = question
        item["ctxs"] = call_retriever(paragraph, [question])

        zero_scroll_items.append(item)

    return zero_scroll_items

def load_scroll(task="qasper", split="validation", kb_only=False, retriever_name="dragon_retriever_chunkbysents300"):

    if task not in ["qasper", "narrative_qa", "quality", "gov_report", "summ_screen_fd", "qmsum"]:
        raise ValueError('invalid task name for scroll"')

    scroll_items = []
    dataset = load_dataset("tau/scrolls", task, cache_dir="./cache")["validation"]
    for i, sample in enumerate(dataset):
        print("sample {}\n".format(i))
        answer = sample["output"]

        if task == "quality":
            items = sample["input"].split("\n\n")
            question = "\n\n".join(items[0:2])
            paragraph = "\n\n".join(items[2:]).strip()
        elif task in ["gov_report", "summ_screen_fd"]:
            paragraph = sample["input"].strip()
            question = None
        elif task in ["qmsum", "narrative_qa", "qasper"]:
            items = sample["input"].split("\n\n")
            question = items[0]
            paragraph = "\n\n".join(items[1:]).strip()
        else:
            raise ValueError("invalid task")

        item = {}
        item["context"] = paragraph
        item["answers"] = [answer]
        item["question"] = question
        scroll_items.append(item)
    
    scroll_final_items = scroll_items
    ## to accelerate the retriever by combining queries for the same context
    if task in ["qmsum", "narrative_qa", "qasper", "quality"]: 
        def merge(temp_item, all_ctxs):
            assert len(all_ctxs) == len(temp_item)
            for item, ctx in zip(temp_item, all_ctxs):
                item["ctxs"] = ctx
            return temp_item
        previous_paragraph = None
        temp_item = []
        quries = []
        scroll_final_items = []
        for idx, item in enumerate(scroll_items):
            if previous_paragraph is None:
                temp_item.append(item)
                quries.append(item["question"])
                previous_paragraph = item["context"]
                continue
            paragraph = item["context"]
            if paragraph == previous_paragraph:
                print("same paragraph")
                temp_item.append(item)
                quries.append(item["question"])
            else:
                print("start retriever")
                all_ctxs = call_retriever([previous_paragraph], quries, retriever_name=retriever_name)
                print(len(quries), len(all_ctxs))
                assert len(all_ctxs) == len(quries)
                assert len(quries) == len(temp_item)
                temp_item = merge(temp_item, all_ctxs)
                scroll_final_items.extend(temp_item)

                previous_paragraph = paragraph
                quries = [item["question"]]
                temp_item = [item]

        ## clear everything in the buffer
        all_ctxs = call_retriever([previous_paragraph], quries, retriever_name=retriever_name)
        temp_item = merge(temp_item, all_ctxs)
        scroll_final_items.extend(temp_item)

    assert len(scroll_final_items) == len(scroll_items)
    return scroll_final_items

def call_retriever(paragraph, queries, retriever_name="dragon_retriever_chunkbysents300"):

    def convert_paragraph_to_kb_format(paragraph):
        return [{"title": None, "text": p} for p in paragraph]

    from evaluate_retriever import all_retrievers, run_retriever, chunk_manual
    rows = convert_paragraph_to_kb_format(paragraph)
    retriever = all_retrievers[retriever_name]
    chunk_args = retriever["chunk_args"] if "chunk_args" in retriever else None
    all_contexts = chunk_manual(rows, chunk_args)
    print("total num of all_contexts", len(all_contexts))
    print("len of queries", len(queries))
    relevant_contexts = run_retriever(retriever, all_contexts, queries)
    print("len of relevant_contexts", len(relevant_contexts))

    relevant_contexts_wo_title = [[item[0] for item in context] for context in relevant_contexts]

    return relevant_contexts_wo_title

def get_zero_scroll_template(task, paragraph, question=None):

    instructions= {}
    instructions["gov_report"] = "You are given a report by a government agency. Write a one-page summary of the report."
    instructions["summ_screen_fd"] = "You are given a script of a TV episode. Summarize the episode in a paragraph."
    instructions["qmsum"] = "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences."
    instructions["squality"] = "" 
    instructions["qasper"] = 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".'
    instructions["narrative_qa"] = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible."
    instructions["quality"] = "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D)."
    instructions["musique"] = ""
    instructions["space_digest"] = ""
    instructions["book_sum_sort"] = ""

    instruction = instructions[task]
    if task == "quality":
        template="{}\n\nStory:\n{}\n\nQuestion and Possible Answers:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
    elif task == "qasper":
        template="{}\n\nArticle:\n{}\n\nQuestion:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
    elif task == "narrative_qa":
        template="{}\n\nStory:\n{}\n\nQuestion:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
    elif task == "musique":
        template="{}\n\n{}\n\nQuestion:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
    elif task == "gov_report":
        template="{}\n\nReport:\n{}\n\nSummary:\n".format(instruction, paragraph)
    elif task == "qmsum":
        template="{}\n\nTranscript:\n{}\n\nQuery:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
    elif task == "summ_screen_fd":
        template="{}\n\nEpisode Script:\n{}\n\nSummary:\n".format(instruction, paragraph)
    else:
        raise ValueError('invalid task name for zero_scroll') # , choose from "qasper", "narrative_qa", "musique", "quality"

    return template

def main(task="mrqa", subset_key="TextbookQA", base_dir="./eval_data/", split="test"):
    
    if task == "mrqa":
        assert split != "train", "no train for TextbookQA"
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

    elif task in ["qasper", "narrative_qa", "quality", "gov_report", "summ_screen_fd", "qmsum"]:
    # elif task in ["qasper", "narrative_qa", "musique", "quality"]:
        assert split == "test"
        # valid_items = load_zero_scroll(task=task)
        retriever_name="dragon_retriever_chunkbysents300"
        valid_items = load_scroll(task=task, retriever_name="dragon_retriever_chunkbysents300")
        subset_key = task + "_" + retriever_name
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
    split = "test"
    # main("boolq", split=split)
    # main("msmarco", split=split)
    # main("race", split=split)
    # main("multirc", split=split)
    # main("mrqa", subset_key="DuoRC.ParaphraseRC", split=split)
    # main("mrqa", subset_key="TextbookQA", split=split)
    # main("mrqa", subset_key="BioASQ", split=split)
    # main("gov_report", split=split, base_dir="./scroll_eval_data/")
    # main("qmsum", split=split, base_dir="./scroll_eval_data/")
    # main("narrative_qa", split=split, base_dir="./scroll_eval_data/")
    # main("qasper", split=split, base_dir="./scroll_eval_data/")
    # main("narrative_qa", split=split, base_dir="./scroll_eval_data/")
    # main("quality", split=split, base_dir="./scroll_eval_data/")
    main("summ_screen_fd", split=split, base_dir="./scroll_eval_data/")
    # main("musique", split=split)
    # main("quality", split=split)
