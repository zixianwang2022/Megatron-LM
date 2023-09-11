
import os
import json
import random


def shuffle_datasets(input_datapath, output_datapath):
    print("shuffle data from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        data_list = json.load(f)
    random.shuffle(data_list)
    
    print("dumping data to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        json.dump(data_list, f)


def count_cannot_answer(input_datapath, num_samples):
    print("reading from %s"  % input_datapath)
    with open(input_datapath, "r") as f:
        data_list = json.load(f)
    
    count = 0
    for item in data_list[:num_samples]:
        if item['answer'] == "Sorry. I cannot find the answer based on the context.":
            count += 1
    
    print("number of cannot answer cases: %d (out of %d)" % (count, num_samples))


if __name__ == "__main__":
    random.seed(1234)
    
    # datafolder_v1 = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa"
    datafolder_v2 = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa"
    
    ### shuffle data (multi-turn)
    # # doc2dial
    # filename = "doc2dial_dragon_QA_test.json"
    # filename = "doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # filename = "doc2dial_e5unsupervised_QA_test.json"
    # filename = "doc2dial_fte5unsupervised_chatgptgen7k_chunk150_QA_test.json"
    # input_datapath = os.path.join(datafolder_v1, "doc2dial/without_shuffle/%s" % filename)
    # output_datapath = os.path.join(datafolder_v1, "doc2dial/%s" % filename)
    # shuffle_datasets(input_datapath, output_datapath)

    # # quac
    # filename = "quac_dragon_QA_test.json"
    # filename = "quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # filename = "quac_e5unsupervised_QA_test.json"
    # filename = "quac_fte5unsupervised_chatgptgen7k_chunk150_QA_test.json"
    # input_datapath = os.path.join(datafolder_v1, "quac/without_shuffle/%s" % filename)
    # output_datapath = os.path.join(datafolder_v1, "quac/%s" % filename)
    # shuffle_datasets(input_datapath, output_datapath)

    # # qrecc
    # filename = "qrecc_dragon_QA_test.json"
    # filename = "qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # filename = "qrecc_e5unsupervised_QA_test.json"
    # filename = "qrecc_fte5unsupervised_chatgptgen7k_chunk150_QA_test.json"
    # input_datapath = os.path.join(datafolder_v1, "qrecc/without_shuffle/%s" % filename)
    # output_datapath = os.path.join(datafolder_v1, "qrecc/%s" % filename)
    # shuffle_datasets(input_datapath, output_datapath)
    
    # # sharc
    # filename = "sharc_dragon_QA_test.json"
    # filename = "sharc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # filename = "sharc_e5unsupervised_QA_test.json"
    # filename = "sharc_fte5unsupervised_chatgptgen7k_chunk150_QA_test.json"
    # input_datapath = os.path.join(datafolder_v1, "sharc/without_shuffle/%s" % filename)
    # output_datapath = os.path.join(datafolder_v1, "sharc/%s" % filename)
    # shuffle_datasets(input_datapath, output_datapath)
    

    ### check cannot answer ratio
    # # filename = "quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # input_datapath = os.path.join(datafolder_v1, "quac/%s" % filename)
    # num_samples = 1000
    # count_cannot_answer(input_datapath, num_samples)

    # filename = "sharc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # input_datapath = os.path.join(datafolder_v1, "sharc/%s" % filename)
    # num_samples = 1000
    # count_cannot_answer(input_datapath, num_samples)


    ### shuffle data (single-turn)
    # # BioASQ
    # filename = "test.json"
    # input_datapath = os.path.join(datafolder_v2, "BioASQ/without_shuffle/%s" % filename)
    # output_datapath = os.path.join(datafolder_v2, "BioASQ/%s" % filename)
    # shuffle_datasets(input_datapath, output_datapath)

    # boolq
    filename = "test.json"
    input_datapath = os.path.join(datafolder_v2, "boolq/without_shuffle/%s" % filename)
    output_datapath = os.path.join(datafolder_v2, "boolq/%s" % filename)
    shuffle_datasets(input_datapath, output_datapath)

    # DuoRC_ParaphraseRC
    filename = "test.json"
    input_datapath = os.path.join(datafolder_v2, "DuoRC_ParaphraseRC/without_shuffle/%s" % filename)
    output_datapath = os.path.join(datafolder_v2, "DuoRC_ParaphraseRC/%s" % filename)
    shuffle_datasets(input_datapath, output_datapath)

    # msmarco
    filename = "test.json"
    input_datapath = os.path.join(datafolder_v2, "msmarco/without_shuffle/%s" % filename)
    output_datapath = os.path.join(datafolder_v2, "msmarco/%s" % filename)
    shuffle_datasets(input_datapath, output_datapath)

    # multirc
    filename = "test.json"
    input_datapath = os.path.join(datafolder_v2, "multirc/without_shuffle/%s" % filename)
    output_datapath = os.path.join(datafolder_v2, "multirc/%s" % filename)
    shuffle_datasets(input_datapath, output_datapath)

    # race
    filename = "test.json"
    input_datapath = os.path.join(datafolder_v2, "race/without_shuffle/%s" % filename)
    output_datapath = os.path.join(datafolder_v2, "race/%s" % filename)
    shuffle_datasets(input_datapath, output_datapath)

    # TextbookQA
    filename = "test.json"
    input_datapath = os.path.join(datafolder_v2, "TextbookQA/without_shuffle/%s" % filename)
    output_datapath = os.path.join(datafolder_v2, "TextbookQA/%s" % filename)
    shuffle_datasets(input_datapath, output_datapath)

