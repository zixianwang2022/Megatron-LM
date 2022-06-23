#!/bin/bash

# provide the MEGATRON API URL
MEGATRON_API='http://10.14.74.235:5000/api' # 530B

# download the data from 

# DATASET_FOLDER=<PATH_OF_YOUR_DATASET_FOLDER>
DATASET_FOLDER=/raid/dansu/datasets/open_domain_data

# set the dataset path
nq_prompt_file=${DATASET_FOLDER}/NQ/train.json
tqa_prompt_file=${DATASET_FOLDER}/TQA/train.json

# You also need to provide the file path to save the prompt data embeddings, 
# so that the retriever will save the embeddings, and load them directly next time for faster retrieval.
nq_encoded_ctx_file=${DATASET_FOLDER}/NQ/encoded_ctx_files_all_multisetdpr_queryctx.pickle
tqa_encoded_ctx_file=${DATASET_FOLDER}/TQA/encoded_ctx_files_all_multisetdpr_queryctx.pickle


python3 ./cgap/api_cgap.py \
        --megatron-api-url ${MEGATRON_API} \
        --nq-prompt-file ${nq_prompt_file} \
        --tqa-prompt-file ${tqa_prompt_file} \
        --nq-encoded-ctx-file ${nq_encoded_ctx_file} \
        --tqa-encoded-ctx-file ${tqa_encoded_ctx_file} \
        --db-name 'NQ' \
        --margin-number 4 \
        --micro-batch-size 4 \
        --ctx-length 128 \
