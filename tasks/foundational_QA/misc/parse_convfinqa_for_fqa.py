
import json


def format_qa_history_to_question(qa_history, num_turn=7, all_turns=False):
    question = ""
    if not all_turns:
        qa_history = qa_history[-num_turn:]
    for item in qa_history:
        if item[0] == "user":
            question += "User: " + item[1] + "\n\n"
        else:
            assert item[0] == "agent"
            question += "Assistant: " + item[1] + "\n\n"

    question += "Assistant:"
    
    return question


def table_row_to_text(header, row):
    '''
    use templates to convert table row to text
    '''
    res = ""
    
    if header[0]:
        res += (header[0] + " ")

    for head, cell in zip(header[1:], row[1:]):
        res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")
    
    res = remove_space(res)
    return res.strip()

def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)

    return " ".join(res)


def parse_convfinqa_data(datapath):
    print("reading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    avg_doc_len = 0
    data_for_fqa_list = []
    for item in data_list:
        pre_text_list = item["pre_text"]
        post_text_list = item["post_text"]

        ## get text before and after Table
        pre_text_list = [text.strip() for text in pre_text_list if len(text.split()) > 3]
        post_text_list = [text.strip() for text in post_text_list if len(text.split()) > 3]

        query_list = item['annotation']['dialogue_break']
        answer_list = item['annotation']['turn_program']

        assert len(query_list) == len(answer_list)

        ## get question_list and answer_list
        qa_history = []
        question_list = []
        for query, answer in zip(query_list, answer_list):
            qa_history.append(("user", query))
            question = format_qa_history_to_question(qa_history)
            question_list.append(question)

            qa_history.append(("agent", answer))
        
        assert len(question_list) == len(answer_list)

        ## process Table
        table = item['table']
        table_text_list = []
        assert len(table) >= 2
        header = table[0]
        for row in table[1:]:
            table_text = table_row_to_text(header, row)
            # print(table_text)
            table_text_list.append(table_text)
        
        chunk_list = pre_text_list + table_text_list + post_text_list
        document = "\n".join(chunk_list)
        avg_doc_len += len(document.split())
        
        for question, answer in zip(question_list, answer_list):
            data_item = {
                "sub-paragraphs": document,
                "question": question,
                "answer": answer
            }
            data_for_fqa_list.append(data_item)

    print("number of dialogs: %d" % len(data_list))
    print("number of turns: %d" % len(data_for_fqa_list))
    print("average document length: %.4f" % (avg_doc_len / len(data_list)))

    return data_for_fqa_list


def parse_finqa_data(datapath):
    print("reading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    num_skip_table = 0
    avg_doc_len = 0
    data_for_fqa_list = []
    for item in data_list:
        pre_text_list = item["pre_text"]
        post_text_list = item["post_text"]

        ## get text before and after Table
        pre_text_list = [text.strip() for text in pre_text_list if len(text.split()) > 3]
        post_text_list = [text.strip() for text in post_text_list if len(text.split()) > 3]

        ## get question and answer
        question = item['qa']['question']
        answer = item['qa']['program']

        ## get table
        table = item['table']
        table_text_list = []

        # assert len(table) >= 2
        if len(table) >= 2:
            header = table[0]
            for row in table[1:]:
                table_text = table_row_to_text(header, row)
                # print(table_text)
                table_text_list.append(table_text)
        
        chunk_list = pre_text_list + table_text_list + post_text_list
        document = "\n".join(chunk_list)
        avg_doc_len += len(document.split())

        data_item = {
            "sub-paragraphs": document,
            "question": question,
            "answer": answer
        }
        data_for_fqa_list.append(data_item)

    print("number of qa pairs: %d" % len(data_for_fqa_list))
    print("average document length: %.4f" % (avg_doc_len / len(data_list)))

    return data_for_fqa_list


def save_fqa_data_list(data_for_fqa_list, output_datapath):
    
    print("writing data_for_fqa_list to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        json.dump(data_for_fqa_list, f, indent=2)


def main_convfinqa():
    train_datapath = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/convfinqa/train.json"
    dev_datapath = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/convfinqa/dev.json"

    data_for_fqa_train_list = parse_convfinqa_data(train_datapath)
    data_for_fqa_dev_list = parse_convfinqa_data(dev_datapath)
    
    output_traindatapath = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/convfinqa/convfinqa_QA_train.json"
    output_devdatapath = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/convfinqa/convfinqa_QA_dev.json"
    save_fqa_data_list(data_for_fqa_train_list, output_traindatapath)
    save_fqa_data_list(data_for_fqa_dev_list, output_devdatapath)


def main_finqa():
    train_datapath = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/finqa/train.json"
    dev_datapath = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/finqa/test.json"

    data_for_fqa_train_list = parse_finqa_data(train_datapath)
    data_for_fqa_dev_list = parse_finqa_data(dev_datapath)

    output_traindatapath = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/finqa/finqa_QA_train.json"
    output_devdatapath = "/lustre/fsw/portfolios/adlr/users/zihanl/datasets/multi-turn-qa/finqa/finqa_QA_dev.json"
    save_fqa_data_list(data_for_fqa_train_list, output_traindatapath)
    save_fqa_data_list(data_for_fqa_dev_list, output_devdatapath)


if __name__ == "__main__":
    # main_convfinqa()
    main_finqa()
