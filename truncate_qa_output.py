#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=False,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=False,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=False,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    group.add_argument('-f', type=str, default='',
                   help='Make jupyter happy')
    args = parser.parse_args()
    args.keep_empty = False

#     if args.tokenizer_type.lower().startswith('bert'):
#         if not args.split_sentences:
#             print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

args = get_args()


# In[4]:


args.tokenizer_type = "GPT2BPETokenizer"
args.vocab_file = "../megatron-lm//gpt2-vocab.json"
args.merge_file = "../megatron-lm/gpt2-merges.txt"

# megatron_tokenizer = build_tokenizer(args)


# In[19]:


# prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_8.3b_test_greedy_0_4000_389532.txt.bak"
# prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_full_8_375000_retro.txt"
# prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_8.3b_test_greedy_concat_1100_389532.txt"
# prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_concatenated_8_375000.txt"

# prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532.concat.txt"
# prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_8_375000.concat.txt"
# prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_tqa_1.3b_test_greedy_0_1100_389532.concat.txt"
# prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_tqa_1.3b_test_greedy_0_1100_8_375000.catcat.txt"


prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_nq_8.3b_test_greedy_0_400_389532_short_format.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_2_375000_short_format.concat.txt"
# prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_tqa_1.3b_test_greedy_0_1100_389532.concat.txt"
# prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_tqa_1.3b_test_greedy_0_1100_8_375000.catcat.txt"


prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_1_375000_1_short_format.txt.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_2_375000_1_short_format.txt.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_1_375000_2_short_format.txt.concat.txt"
prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_2_375000_2_short_format.txt.concat.txt"

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_3_375000_2_short_format.txt.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_4_375000_2_short_format.txt.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_5_375000_2_short_format.txt.concat.txt"
prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_nq_8.3b_test_greedy_0_400_6_375000_2_short_format.txt.concat.txt"

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_6_375000_2_short_format.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_2_short_format.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_4_375000_2_short_format.concat.txt"
prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_8_375000_2_short_format.concat.txt"

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_3_short_format.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_4_short_format.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_5_short_format.concat.txt"
prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-8.3b-pretraining-retro-K-2/generate_tqa_8.3b_test_greedy_0_1100_3_375000_6_short_format.concat.txt"

# prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_nq_8.3b_test_greedy_0_400_389532_short_format_4.concat.txt"
# prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_nq_8.3b_test_greedy_0_400_389532_short_format_5.concat.txt"
# prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_nq_8.3b_test_greedy_0_400_389532_short_format_6.concat.txt"
#
# prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_4.concat.txt"
# prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_5.concat.txt"
# prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-8.3b/generate_tqa_8.3b_test_greedy_0_1100_389532_short_format_6.concat.txt"
# In[25]:

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_tqa_8.3b_test_greedy_0_1100_3_80000_3_short_format.txt.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_nq_8.3b_test_greedy_0_400_2_80000_2_short_format.txt.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_tqa_8.3b_test_greedy_0_1100_3_60000_3_short_format.txt.concat.txt"
prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_nq_8.3b_test_greedy_0_400_2_60000_2_short_format.txt.concat.txt"

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_tqa_8.3b_test_greedy_0_1100_3_50000_3_short_format.txt.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_nq_8.3b_test_greedy_0_400_2_50000_2_short_format.txt.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_tqa_8.3b_test_greedy_0_1100_3_50000_3_short_format.txt.concat.txt"
prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_nq_8.3b_test_greedy_0_400_2_50000_2_short_format.txt.concat.txt"

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_tqa_8.3b_test_greedy_0_1100_3_100000_3_short_format.txt.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-5/generate_nq_8.3b_test_greedy_0_400_2_100000_2_short_format.txt.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_tqa_8.3b_test_greedy_0_1100_3_100000_3_short_format.txt.concat.txt"
prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/gpt3-8.3b-pretraining-retro-fitting-K-2-lr-1e-6/generate_nq_8.3b_test_greedy_0_400_2_100000_2_short_format.txt.concat.txt"

prediction_files = [prediction_file1,prediction_file2,prediction_file3,prediction_file4]


prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr/generate_843m_test_greedy_0_400_2835248.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting/generate_843m_test_greedy_0_400_194000.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting/generate_843m_test_greedy_0_400_195312.concat.txt"
prediction_files = [prediction_file1,prediction_file2,prediction_file3]



# prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_2_375000_1_short_format.concat.txt"
# prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532_short_format_1.concat.txt"
#
# prediction_files = [prediction_file1,prediction_file2,prediction_file3]
#
prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc/generate_2b_test_greedy_0_400_1417624.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting/generate_2b_test_greedy_0_400_97656.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting/generate_2b_test_greedy_0_400_97656.concat.txt"
prediction_files = [prediction_file1,prediction_file2,prediction_file3]

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc/generate_8b_test_greedy_0_400_1417624.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting/generate_8b_test_greedy_0_400_97656.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/generate_8b_test_greedy_0_400_97656.concat.txt"
prediction_files = [prediction_file1,prediction_file2,prediction_file3]

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc/generate_22b_test_greedy_0_400_708812.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting/generate_22b_test_greedy_0_400_48828.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/generate_22b_test_greedy_0_400_48828.concat.txt"
prediction_files = [prediction_file1,prediction_file2,prediction_file3]

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1/generate_43b_test_greedy_0_400_472541.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1/generate_43b_test_greedy_0_400_32552.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_32552.concat.txt"
prediction_files = [prediction_file1,prediction_file2,prediction_file3]

prediction_file1 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_27000.concat.txt"
prediction_file2 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_28000.concat.txt"
prediction_file3 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_29000.concat.txt"
prediction_file4 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_30000.concat.txt"
prediction_file5 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_31000.concat.txt"
prediction_file6 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_32000.concat.txt"
prediction_file7 = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_nq_43b_test_greedy_0_400_32552.concat.txt"
prediction_files = [prediction_file1,prediction_file2,prediction_file3,prediction_file4,prediction_file5,prediction_file6,prediction_file7]


prediction_files = []
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr/generate_tqa_843m_test_greedy_0_1100_2835248.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting/generate_tqa_843m_test_greedy_0_1100_194000.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting/generate_tqa_843m_test_greedy_0_1100_195312.concat.txt")

prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc/generate_tqa_2b_test_greedy_0_1100_1417624.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting/generate_tqa_2b_test_greedy_0_1100_97656.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting/generate_tqa_2b_test_greedy_0_1100_97656.concat.txt")

prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc/generate_tqa_8b_test_greedy_0_1100_1417624.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting/generate_tqa_8b_test_greedy_0_1100_97656.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/generate_tqa_8b_test_greedy_0_1100_97656.concat.txt")

prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc/generate_tqa_22b_test_greedy_0_1100_708812.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting/generate_tqa_22b_test_greedy_0_1100_48828.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/generate_tqa_22b_test_greedy_0_550_48828.concat.txt")

prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1/generate_tqa_43b_test_greedy_0_550_472541.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1/generate_tqa_43b_test_greedy_0_550_32552.concat.txt")
prediction_files.append("/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_tqa_43b_test_greedy_0_400_32000.concat.txt")

prediction_files = []
model_name = "gpt3-43b-multi-1.1t-gtc/tp8pp1"
ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/{}/".format(model_name)
prediction_files.append(ckpt_path + "/generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_472541.txt")
prediction_files.append(ckpt_path + "generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt")
prediction_files.append(ckpt_path + "/generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt")
prediction_files.append(ckpt_path + "generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_472541.txt")
prediction_files.append(ckpt_path + "/generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_472541.txt")
prediction_files.append(ckpt_path + "/generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_472541.txt")
prediction_files.append(ckpt_path + "/generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt")
prediction_files.append(ckpt_path + "/generate_nq_43b_test_greedy_0_200_472541.txt")
# In[7]:

prediction_files = []
model_name = "gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed"
model_name = "gpt3-43b-pretraining-gpt-fitting-tp8pp1"
ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/{}/".format(model_name)
prediction_files.append(ckpt_path + "/generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_32552.txt")
prediction_files.append(ckpt_path + "generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt")
prediction_files.append(ckpt_path + "/generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt")
prediction_files.append(ckpt_path + "generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_32552.txt")
prediction_files.append(ckpt_path + "/generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt")
prediction_files.append(ckpt_path + "/generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt")
prediction_files.append(ckpt_path + "/generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt")
prediction_files.append(ckpt_path + "/generate_nq_43b_test_greedy_0_200_32552.txt")

# In[15]:


# In[8]:


# In[11]:




# In[12]:



def truncate_32(prediction_file):
    with open(prediction_file) as f:
        lines = f.readlines()
    print(len(lines))    
    tokens = [megatron_tokenizer.tokenize(line) for line in lines]    
    import numpy as np
    print(np.mean([len(token) for token in tokens]))
    truncated_tokens = [token[:32] for token in tokens]    
    new_lines = [megatron_tokenizer.detokenize(token) for token in truncated_tokens]

    with open(prediction_file + ".truncate32.txt", "w") as f:
        for line in new_lines:
            line = line[:line.find("<|endoftext|>")].strip().replace("\n", " ")
            f.write(line + '\n')
    print(prediction_file + ".truncate32.txt")


def truncate_20(prediction_file):
    with open(prediction_file) as f:
        lines = f.readlines()
    print(len(lines))    
    tokens = [megatron_tokenizer.tokenize(line) for line in lines]    
    import numpy as np
    print(np.mean([len(token) for token in tokens]))
    truncated_tokens = [token[:20] for token in tokens]    
    new_lines = [megatron_tokenizer.detokenize(token) for token in truncated_tokens]

    with open(prediction_file + ".truncate20.txt", "w") as f:
        for line in new_lines:
            line = line[:line.find("<|endoftext|>")].strip().replace("\n", " ")
            f.write(line + '\n')
    print(prediction_file + ".truncate20.txt")


# In[24]:


def truncate_10(prediction_file):
    with open(prediction_file) as f:
        lines = f.readlines()
    print(len(lines))    
    tokens = [megatron_tokenizer.tokenize(line) for line in lines]    
    import numpy as np
    print(np.mean([len(token) for token in tokens]))
    truncated_tokens = [token[:10] for token in tokens]    
    new_lines = [megatron_tokenizer.detokenize(token) for token in truncated_tokens]

    with open(prediction_file + ".truncate10.txt", "w") as f:
        for line in new_lines:
            line = line[:line.find("<|endoftext|>")].strip().replace("\n", " ")
            f.write(line + '\n')
    print(prediction_file + ".truncate10.txt")


# In[26]:

def truncate_period(prediction_file):
    with open(prediction_file) as f:
        lines = f.readlines()
    print(len(lines))

    with open(prediction_file + ".period.txt", "w") as f:
        for line in lines:
            line = line[:line.find(".")].strip().replace("\n", " ")
            f.write(line + '\n')
    print(prediction_file + ".period.txt")

for f in prediction_files:
    # truncate_32(f)
    # truncate_20(f)
    # truncate_10(f)
    truncate_period(f)


# In[ ]:




