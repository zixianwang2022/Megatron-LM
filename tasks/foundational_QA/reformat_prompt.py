# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def reformat_prompt_for_malachite_sawfly(query, neighbours, dataset_name, ft_neighbours, \
            max_output_len, tokenizer, max_seq_length):

    system = "<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq", "BioASQ", "DuoRC_ParaphraseRC", "TextbookQA"]
    yes_no_without_context = ["boolq", "multirc"]
    multichoices = ["race"]
    # multi-turn qa datasets
    formatted_dataset_name = ["convqa", "chatgptgen", "doc2dial", "quac", "qrecc", "sharc"]
    user_template = ""

    if dataset_name in formatted_dataset_name:
        # Add <extra_id_1> in front of User and Assistant
        query = query.replace("User: ", "<extra_id_1>User\n")
        query = query.replace("Assistant: ", "<extra_id_1>Assistant\n")
        # for the last "Assistant:"
        query = query.replace("Assistant:", "<extra_id_1>Assistant\n")
        query = query.replace("\n\n", "\n")

        dialogue_turn = query
    else:
        if dataset_name in short_span_with_context:
            user = "Answer the following question with a short span. {}".format(query)
        elif dataset_name in yes_no_without_context:
            user = "Answer the following question with True or False. {}".format(query)
        elif dataset_name in multichoices:
            user = "Answer the following question by selecting one of the provided options. {}".format(query)
        else:
            user = "Please give a full and complete answer for the question. {}".format(query)

        dialogue_format="<extra_id_1>User\n{}\n<extra_id_1>Assistant\n"
        dialogue_turn = dialogue_format.format(user)
    
    steering_instruction = "<extra_id_2>quality:4,toxicity:0,violence:0,helpfulness:4,not_appropriate:0,hate_speech:0,sexual_content:0,fails_task:0,political_content:0,moral_judgement:0,lang:en\n"

    if ft_neighbours > 0:

        context = "\n\n".join(neighbours[0:ft_neighbours]) + "\n\n"
        context_tokens = tokenizer.tokenize(context)
        dialogue_tokens = tokenizer.tokenize(dialogue_turn)
        system_tokens = tokenizer.tokenize(system)
        instruction_tokens = tokenizer.tokenize(steering_instruction)
        context_tokens = context_tokens[:max_seq_length - max_output_len - len(dialogue_tokens) - len(system_tokens) - len(instruction_tokens)]
        context = tokenizer.detokenize(context_tokens)

        all_input = system + context + dialogue_turn + steering_instruction
        input_tokens = tokenizer.tokenize(all_input)

    else:
        all_input = system + dialogue_turn + steering_instruction
        input_tokens = tokenizer.tokenize(all_input)
    
    return  input_tokens


def reformat_prompt_for_neat_spoonbill(query, neighbours, dataset_name, ft_neighbours, \
            max_output_len, tokenizer, max_seq_length):

    system = "<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq", "BioASQ", "DuoRC_ParaphraseRC", "TextbookQA"]
    yes_no_without_context = ["boolq", "multirc"]
    multichoices = ["race"]
    # multi-turn qa datasets
    formatted_dataset_name = ["convqa", "chatgptgen", "doc2dial", "quac", "qrecc", "sharc"]
    user_template = ""

    if dataset_name in formatted_dataset_name:
        # Add <extra_id_1> in front of User and Assistant
        query = query.replace("User: ", "<extra_id_1>User\n")
        query = query.replace("Assistant: ", "<extra_id_1>Assistant\n")
        # for the last "Assistant:"
        query = query.replace("Assistant:", "<extra_id_1>Assistant\n")
        query = query.replace("\n\n", "\n")

        dialogue_turn = query
    else:
        if dataset_name in short_span_with_context:
            user = "Answer the following question with a short span. {}".format(query)
        elif dataset_name in yes_no_without_context:
            user = "Answer the following question with True or False. {}".format(query)
        elif dataset_name in multichoices:
            user = "Answer the following question by selecting one of the provided options. {}".format(query)
        else:
            user = "Please give a full and complete answer for the question. {}".format(query)

        dialogue_format="<extra_id_1>User\n{}\n<extra_id_1>Assistant\n"
        dialogue_turn = dialogue_format.format(user)
    
    steering_instruction = "<extra_id_2>quality:9,toxicity:0,violence:0,helpfulness:9,not_appropriate:0\n"

    if ft_neighbours > 0:

        context = "\n\n".join(neighbours[0:ft_neighbours]) + "\n\n"
        context_tokens = tokenizer.tokenize(context)
        dialogue_tokens = tokenizer.tokenize(dialogue_turn)
        system_tokens = tokenizer.tokenize(system)
        instruction_tokens = tokenizer.tokenize(steering_instruction)
        context_tokens = context_tokens[:max_seq_length - max_output_len - len(dialogue_tokens) - len(system_tokens) - len(instruction_tokens)]
        context = tokenizer.detokenize(context_tokens)

        all_input = system + context + dialogue_turn + steering_instruction
        input_tokens = tokenizer.tokenize(all_input)

    else:
        all_input = system + dialogue_turn + steering_instruction
        input_tokens = tokenizer.tokenize(all_input)
    
    return  input_tokens


def reformat_prompt_for_rlhf_models(query, neighbours, dataset_name, ft_neighbours, \
            max_output_len, tokenizer, max_seq_length):

    system = "<extra_id_0>System\n\n"

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq", "BioASQ", "DuoRC_ParaphraseRC", "TextbookQA"]
    yes_no_without_context = ["boolq", "multirc"]
    multichoices = ["race"]
    # multi-turn qa datasets
    formatted_dataset_name = ["convqa", "chatgptgen", "doc2dial", "quac", "qrecc", "sharc"]
    user_template = ""

    if dataset_name in formatted_dataset_name:
        # Add <extra_id_1> in front of User and Assistant
        query = query.replace("User: ", "<extra_id_1>User\n")
        query = query.replace("Assistant: ", "<extra_id_1>Assistant\n")
        # for the last "Assistant:"
        query = query.replace("Assistant:", "<extra_id_1>Assistant\n")
        query = query.replace("\n\n", "\n")

        dialogue_turn = query
    else:
        if dataset_name in short_span_with_context:
            user = "Answer the following question with a short span. {}".format(query)
        elif dataset_name in yes_no_without_context:
            user = "Answer the following question with True or False. {}".format(query)
        elif dataset_name in multichoices:
            user = "Answer the following question by selecting one of the provided options. {}".format(query)
        else:
            user = "Please give a full and complete answer for the question. {}".format(query)

        dialogue_format="<extra_id_1>User\n{}\n<extra_id_1>Assistant\n"
        dialogue_turn = dialogue_format.format(user)
    
    if ft_neighbours > 0:

        context = "\n\n".join(neighbours[0:ft_neighbours]) + "\n\n"
        context_tokens = tokenizer.tokenize(context)
        dialogue_tokens = tokenizer.tokenize(dialogue_turn)
        system_tokens = tokenizer.tokenize(system)
        context_tokens = context_tokens[:max_seq_length - max_output_len - len(dialogue_tokens) - len(system_tokens)]
        context = tokenizer.detokenize(context_tokens)

        all_input = system + context + dialogue_turn
        input_tokens = tokenizer.tokenize(all_input)

    else:
        all_input = system + dialogue_turn
        input_tokens = tokenizer.tokenize(all_input)
    
    return  input_tokens
