
import json

def get_chatgptgen_subset(input_datapath, output_datapath, num_turns, num_no_answer):
    
    print("loading data from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        data_list = json.load(f)
    subset_data_list = data_list[:num_turns] + data_list[-num_no_answer:]
    
    print("writing data to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        json.dump(subset_data_list, f, indent=2)


if __name__ == "__main__":
    
    ## chatgpt-gen (7k dialogs -> ~30k turns)
    input_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/chatgptgennoanswer/chatgptgennoanswer_QA_train.json"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/chatgptgen3k/chatgptgen3k_QA_train.json"
    # num_turns = 13000
    # num_no_answer = 400
    output_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/chatgptgen1k/chatgptgen1k_QA_train.json"
    num_turns = 4300
    num_no_answer = 200
    get_chatgptgen_subset(input_datapath, output_datapath, num_turns, num_no_answer)
