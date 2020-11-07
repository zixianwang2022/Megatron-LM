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

"""SQuAD dataset."""

import csv
import json
import os
import sys
import time
import torch
from torch.utils.data import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "../../")))
from megatron import print_rank_0
from tasks.t5_model_utils.data_utils import build_sample
from tasks.t5_model_utils.data_utils import build_tokens_types_paddings_from_text

class SQuADDataset(Dataset):
    """SQuAD base dataset class."""

    # Use the max number of references to 6 per question as per SQuAD 1.1 dataset
    def __init__(self, dataset_name, datapaths,
                 tokenizer, max_seq_length, decoder_seq_length, max_refs_count=6):

        # Store inputs.
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.max_refs_count = max_refs_count
        print_rank_0(' > building SQuAD dataset for {}:'.format(self.dataset_name))

        # Process the files.
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)

        self.samples = []        
        for datapath in datapaths:
            self.samples.extend(process_single_datapath(self.dataset_name,
                        datapath, tokenizer, max_seq_length, self.max_refs_count))
        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        raw_sample_in_format = "[question] " + raw_sample['question'] + \
                                " [context] " + raw_sample['context']
        raw_labels = raw_sample['answer']
        raw_references = raw_sample['references']
        enc_ids, tokentypes_enc, dec_in_ids, \
        dec_out_ids, loss_mask = \
            build_tokens_types_paddings_from_text(
            raw_sample_in_format, raw_labels, 
            self.tokenizer, self.max_seq_length,
            self.decoder_seq_length)
        sample = build_sample(enc_ids, tokentypes_enc,
                              dec_in_ids, dec_out_ids,
                              loss_mask, raw_references)
        return sample

def clean_text_squad(tokenizer, input_text):
    """Clean the text as per the given tokenizer guideline used for training"""
    toks = tokenizer.tokenizer.basic_tokenizer.tokenize(input_text)
    return tokenizer.tokenizer.convert_tokens_to_string(toks)

def process_single_datapath(dataset_name, filename, tokenizer, max_seq_length,
                                max_refs_count):
    """Read in SQuAD file, clean-up, tokenize, and convert to
    samples."""

    print_rank_0('   > working on {}'.format(filename))
    start_time = time.time()

    samples = []
    num_contexts = 0
    num_samples = 0

    #Load te file
    with open(filename, "r", encoding='utf-8') as reader:
        data = json.load(reader)["data"]

        #Iterate over the dataset
        for entry in data:
            for paragraph in entry["paragraphs"]:
                # Context text and convert to ids with prefix [CONTEXT]
                context = paragraph["context"]
                num_contexts += 1

                #Loop over the question-answers
                for qas in paragraph["qas"]:
                    # Question and answer
                    qas_id = qas["id"]

                    # Question
                    question = qas["question"]
                    # Answer - Taking the first for training
                    answer = qas["answers"][0]["text"]

                    # Taking multiples answers for evaluation
                    references = []
                    if "validation" in dataset_name:
                        for index in range(len(qas["answers"])):
                            ref = clean_text_squad(tokenizer,
                                    qas["answers"][index]["text"])
                            references.append(ref)

                        if len(references) < max_refs_count:
                            last_ref = references[len(references) - 1]
                            references += (max_refs_count - \
                                len(references)) * [last_ref]

                    # Make a sample and append
                    sample = {'context': context, 'question': question,
                                'answer': answer, 'references': references}
                    samples.append(sample)
                    num_samples += 1

    elapsed_time = time.time() - start_time
    print_rank_0('    > processed contexts {} samples {}'
                 ' in {:.2f} seconds'.format(num_contexts, num_samples,
                 elapsed_time))

    return samples
