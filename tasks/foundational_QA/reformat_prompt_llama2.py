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


def reformat_prompt_llama2_chat(query, neighbours, dataset_name, ft_neighbours, \
    max_output_len, tokenizer, max_seq_length):

    # system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.\n\n"

    if dataset_name in ["oasst", "quiet_cockatoo"]:
        input_tokens = tokenizer.tokenize(system + query)
        # print(dataset_name, system + query)
        return input_tokens

    short_span_with_context = ["drop", "NarrativeQA", "QASC", "Quoref", "ROPES", "squad1.1", "squad2.0", "newsqa", "nq", "BioASQ", "DuoRC_ParaphraseRC", "TextbookQA"]
    yes_no_without_context = ["boolq", "multirc"]
    multichoices = ["race"]
    # multi-turn qa datasets
    formatted_dataset_name = ["convqa", "chatgptgen", "doc2dial", "doc2dialv2", "quac", "quacv2", "qrecc", "sharc", "nvolvemultiturn600"]
    user_template = ""

    if dataset_name in formatted_dataset_name:
        dialogue_turn = query
    else:
        if dataset_name in short_span_with_context:
            # user = "Answer the following question with a few words. {}".format(query)
            user = "Answer the following question with a short span. The answer needs to be just in a few words. {}".format(query)
        elif dataset_name in yes_no_without_context:
            user = "Answer the following question with True or False. {}".format(query)
        elif dataset_name in multichoices:
            user = "Answer the following question by selecting one of the provided options. {}".format(query)
        else:
            user = "Answer the following question with one or two sentences. {}".format(query)

        dialogue_format="User: {}\n\nAssistant:"
        dialogue_turn = dialogue_format.format(user)

    if ft_neighbours > 0:
        # if shuffle_topn:
        #     import random
        #     random.seed(1234)
        #     random_neighbours = neighbours[0:ft_neighbours]
        #     random.shuffle(random_neighbours)
        #     neighbours = random_neighbours + neighbours[ft_neighbours:]
        # Truncate to `max_sequence_length` to fit in output tokens.
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

    # print(dataset_name, all_input)

    return  input_tokens

