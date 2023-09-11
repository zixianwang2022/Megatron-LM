
import json
import random
import os

def select_nv_benefits(input_datapath, output_testpath, output_fewshotpath, n_shot=5):
    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        data_list = json.load(f)

    random.shuffle(data_list)
    
    fewshot_list = data_list[:n_shot]
    fewshot_qa_pairs = []
    for item in fewshot_list:
        question = item['question']
        answer = item['answer']
        fewshot_qa_pairs.append({"question": question, "answer": answer})
    output_list = data_list[n_shot:]
    
    print("writing output_list to %s" % output_testpath)
    with open(output_testpath, "w") as f:
        json.dump(output_list, f, indent=2)
    
    print("writing fewshot_qa_pairs to %s" % output_fewshotpath)
    with open(output_fewshotpath, "w") as f:
        json.dump(fewshot_qa_pairs, f, indent=2)


def select_general_dataests_v2(input_datapath, output_fewshotpath, n_test=250, n_shot=5):
    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        data_list = json.load(f)

    # this data_list is not used for test
    print("total samples: %d; test samples: %d; random select from the rest of %d samples" % \
                                (len(data_list), n_test, len(data_list)-n_test))
    data_list_notest = data_list[n_test:]
    filtered_data_list = []
    noanswer_sent = "Sorry. I cannot find the answer based on the context."
    
    for item in data_list_notest:
        if "answer" in item:
            answer = item['answer']
        else:
            answer = item['answers'][0]
        if noanswer_sent not in item['question'] and noanswer_sent != answer:
            filtered_data_list.append(item)

    random.shuffle(filtered_data_list)
    fewshot_list = filtered_data_list[:n_shot]
    fewshot_qa_pairs = []
    for item in fewshot_list:
        question = item['question']
        if "answer" in item:
            answer = item['answer']
        else:
            answer = item['answers'][0]
        fewshot_qa_pairs.append({"question": question, "answer": answer})

    print("writing fewshot_qa_pairs to %s" % output_fewshotpath)
    with open(output_fewshotpath, "w") as f:
        json.dump(fewshot_qa_pairs, f, indent=2)


def select_general_datasets(input_datapath, output_fewshotpath, n_shot=5):
    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        data_list = json.load(f)

    # remove noanswer case (since no context will be provided in the fewshot samples, no point to provide noanswer case)
    filtered_data_list = []
    noanswer_sent = "Sorry. I cannot find the answer based on the context."
    for item in data_list:
        if "answer" in item:
            answer = item['answer']
        else:
            if len(item['answers']) > 0:
                answer = item['answers'][0]
            else:
                continue
            
        if noanswer_sent not in item['question'] and noanswer_sent != answer:
            filtered_data_list.append(item)

    random.shuffle(filtered_data_list)
    fewshot_list = filtered_data_list[:n_shot]
    fewshot_qa_pairs = []
    for item in fewshot_list:
        question = item['question']
        if "answer" in item:
            answer = item['answer']
        else:
            answer = item['answers'][0]
        if type(answer) == dict:
            answer = answer['text']
        fewshot_qa_pairs.append({"question": question, "answer": answer})

    print("writing fewshot_qa_pairs to %s" % output_fewshotpath)
    with open(output_fewshotpath, "w") as f:
        json.dump(fewshot_qa_pairs, f, indent=2)


if __name__ == "__main__":
    
    data_folder_v1 = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa"
    data_folder_v2 = "/lustre/fsw/adlr/adlr-nlp/pengx"
    data_folder_v3 = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa"
    data_folder_v4 = "/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data"
    
    ### for evaluation data
    # ## NV Benefits
    # random.seed(1234)
    # nv_benefits_input_datapath = "nv_benefits_dragon_retriever300_retrieved_generic/all.json"
    # nv_benefits_output_testpath = "nv_benefits_dragon_retriever300_retrieved_generic/test.json"
    # nv_benefits_output_fewshotpath = "nv_benefits_dragon_retriever300_retrieved_generic/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, nv_benefits_input_datapath)
    # output_testpath = os.path.join(data_folder_v1, nv_benefits_output_testpath)
    # output_fewshotpath = os.path.join(data_folder_v1, nv_benefits_output_fewshotpath)
    # select_nv_benefits(input_datapath, output_testpath, output_fewshotpath)

    # ## Iternal
    # random.seed(1234)
    # iternal_input_datapath = "iternal/test.json"
    # iternal_output_fewshotpath = "iternal/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, iternal_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, iternal_output_fewshotpath)
    # select_general_dataests_v2(input_datapath, output_fewshotpath, n_test=250)

    # ## nq
    # random.seed(123)
    # nq_input_datapath = "retro/data/NQ/train.json"
    # nq_output_fewshotpath = "NQ/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v2, nq_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, nq_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## att
    # random.seed(456)
    # att_input_datapath = "data/att/att_tasb_msmarcominilm_chunkbysents300_retrieved/train.json"
    # att_output_datapath = "att/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v2, att_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, att_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## ford
    # random.seed(1234)
    # ford_input_datapath = "retro/data/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/train.json"
    # ford_output_datapath = "ford/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v2, ford_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, ford_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## nvit
    # random.seed(123456)
    # nvit_input_datapath = "retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/train.json"
    # nvit_output_datapath = "nvit/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v2, nvit_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, nvit_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## landrover
    # random.seed(123)
    # landrover_input_datapath = "retro/data/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/train.json"
    # landrover_output_datapath = "landrover/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v2, landrover_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, landrover_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## doc2dial
    # random.seed(123)
    # doc2dial_input_datapath = "doc2dial/without_shuffle/doc2dial_QA_train.json"
    # doc2dial_output_fewshotpath = "doc2dial/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v3, doc2dial_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v3, doc2dial_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## quac
    # random.seed(789)
    # quac_input_datapath = "quac/without_shuffle/quac_QA_train.json"
    # quac_output_fewshotpath = "quac/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v3, quac_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v3, quac_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## qrecc
    # random.seed(789)
    # qrecc_input_datapath = "qrecc/without_shuffle/qrecc_QA_train.json"
    # qrecc_output_fewshotpath = "qrecc/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v3, qrecc_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v3, qrecc_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## sharc
    # random.seed(12345678)
    # sharc_input_datapath = "sharc/sharc_dragon_QA_test.json"
    # sharc_output_fewshotpath = "sharc/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v3, sharc_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v3, sharc_output_fewshotpath)
    # select_general_dataests_v2(input_datapath, output_fewshotpath, n_test=10000)

    
    # ## BioASQ
    # random.seed(12345)
    # BioASQ_input_datapath = "BioASQ/test.json"
    # BioASQ_output_fewshotpath = "BioASQ/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, BioASQ_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, BioASQ_output_fewshotpath)
    # select_general_dataests_v2(input_datapath, output_fewshotpath, n_test=1000)

    # ## DuoRC_ParaphraseRC
    # random.seed(12345678)
    # DuoRC_ParaphraseRC_input_datapath = "DuoRC_ParaphraseRC/test.json"
    # DuoRC_ParaphraseRC_output_fewshotpath = "DuoRC_ParaphraseRC/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, DuoRC_ParaphraseRC_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, DuoRC_ParaphraseRC_output_fewshotpath)
    # select_general_dataests_v2(input_datapath, output_fewshotpath, n_test=1000)

    # ## boolq
    # random.seed(12345678)
    # boolq_input_datapath = "boolq/without_shuffle/train.json"
    # boolq_output_fewshotpath = "boolq/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, boolq_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, boolq_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## msmarco
    # random.seed(12345678)
    # msmarco_input_datapath = "msmarco/without_shuffle/train.json"
    # msmarco_output_fewshotpath = "msmarco/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, msmarco_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, msmarco_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)
    
    # ## multirc
    # random.seed(123456)
    # multirc_input_datapath = "multirc/without_shuffle/train.json"
    # multirc_output_fewshotpath = "multirc/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, multirc_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, multirc_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)
    
    # ## race
    # random.seed(123456)
    # race_input_datapath = "race/without_shuffle/train.json"
    # race_output_fewshotpath = "race/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, race_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, race_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)
    
    # ## TextbookQA
    # random.seed(123456)
    # TextbookQA_input_datapath = "TextbookQA/test.json"
    # TextbookQA_output_fewshotpath = "TextbookQA/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v1, TextbookQA_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, TextbookQA_output_fewshotpath)
    # select_general_dataests_v2(input_datapath, output_fewshotpath, n_test=1000)
    

    ### for training data
    # ## drop
    # random.seed(123456)
    # drop_input_datapath = "drop/drop_QA_dev.json"
    # drop_output_datapath = "drop/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v4, drop_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, drop_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath, n_shot=16)

    # ## NarrativeQA
    # random.seed(1234567)
    # NarrativeQA_input_datapath = "NarrativeQA/NarrativeQA_QA_dev.json"
    # NarrativeQA_output_datapath = "NarrativeQA/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v4, NarrativeQA_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, NarrativeQA_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath, n_shot=16)

    # ## Quoref
    # random.seed(12345)
    # Quoref_input_datapath = "Quoref/Quoref_QA_dev.json"
    # Quoref_output_datapath = "Quoref/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v4, Quoref_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, Quoref_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath, n_shot=16)

    # ## ROPES
    # random.seed(123)
    # ROPES_input_datapath = "ROPES/ROPES_QA_dev.json"
    # ROPES_output_datapath = "ROPES/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v4, ROPES_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, ROPES_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath, n_shot=16)

    # ## squad1.1
    # random.seed(1234567)
    # squad1_input_datapath = "squad1.1/squad1.1_QA_dev.json"
    # squad1_output_datapath = "squad1.1/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v4, squad1_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, squad1_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath, n_shot=16)

    # ## squad2.0
    # random.seed(1234)
    # squad2_input_datapath = "squad2.0/squad2.0_QA_dev.json"
    # squad2_output_datapath = "squad2.0/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v4, squad2_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, squad2_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath, n_shot=16)

    # ## newsqa
    # random.seed(123456)
    # newsqa_input_datapath = "newsqa/newsqa_QA_dev.json"
    # newsqa_output_datapath = "newsqa/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v4, newsqa_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v1, newsqa_output_datapath)
    # select_general_datasets(input_datapath, output_fewshotpath, n_shot=16)

    ## quiet_cockatoo
    random.seed(456)
    quiet_cockatoo_input_datapath = "quiet_cockatoo/quiet_cockatoo_QA_dev.json"
    quiet_cockatoo_output_datapath = "quiet_cockatoo/fewshot_samples.json"
    input_datapath = os.path.join(data_folder_v4, quiet_cockatoo_input_datapath)
    output_fewshotpath = os.path.join(data_folder_v1, quiet_cockatoo_output_datapath)
    select_general_datasets(input_datapath, output_fewshotpath, n_shot=16)

    # ## convqa
    # random.seed(789)
    # convqa_input_datapath = "convqa/convqa_QA_dev.json"
    # convqa_output_fewshotpath = "convqa/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v3, convqa_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v3, convqa_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)

    # ## chatgptgen
    # random.seed(456789)
    # chatgptgen_input_datapath = "chatgptgen/chatgptgen_QA_dev.json"
    # chatgptgen_output_fewshotpath = "chatgptgen/fewshot_samples.json"
    # input_datapath = os.path.join(data_folder_v3, chatgptgen_input_datapath)
    # output_fewshotpath = os.path.join(data_folder_v3, chatgptgen_output_fewshotpath)
    # select_general_datasets(input_datapath, output_fewshotpath)

