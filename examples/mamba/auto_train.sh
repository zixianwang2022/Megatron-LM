#!/bin/bash

# DEBUG ONLY: 
# current_datetime=$(date +"%Y%m%d_%H%M%S")
# DATASET_SIZE=100
# ./train_debug.sh ./training_${DATASET_SIZE}_question_prompt_document /workspace/data/ssm-retrieval/mamba2-8b/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model > debug_output.txt # _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_output.txt
# mv communication_output.txt _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_communication_output.txt



# Store the current time in a variable
# current_datetime=$(date +"%Y%m%d_%H%M%S")
# DATASET_SIZE=10000
# ./train_without_soup.sh ./training_${DATASET_SIZE}_D_01_Q_A_D_01_Q_A_document /workspace/data/ssm-retrieval/mamba2-8b/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model > output.txt # _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_output.txt
# mv communication_output.txt _${current_datetime}_D_01_Q_A_training_${DATASET_SIZE}_communication_output.txt


# Store the current time in a variable
# current_datetime=$(date +"%Y%m%d_%H%M%S")
# DATASET_SIZE=10000
# ./train.sh ./training_${DATASET_SIZE}_question_prompt_document /workspace/data/ssm-retrieval/mamba2-8b/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model > output.txt # _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_output.txt
# mv communication_output.txt _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_communication_output.txt


# # Store the current time in a variable
# current_datetime=$(date +"%Y%m%d_%H%M%S")
# DATASET_SIZE=10000
# ./train_2.sh ./training_${DATASET_SIZE}_question_prompt_document /workspace/data/ssm-retrieval/mamba2-8b/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model > _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_output.txt
# mv communication_output.txt _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_communication_output.txt


# # # Store the current time in a variable
current_datetime=$(date +"%Y%m%d_%H%M%S")
DATASET_SIZE=10000
./train_3.sh ./training_${DATASET_SIZE}_question_prompt_document /workspace/data/ssm-retrieval/mamba2-8b/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model > _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_output.txt
mv communication_output.txt _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_communication_output.txt


# # # Store the current time in a variable
# current_datetime=$(date +"%Y%m%d_%H%M%S")
# DATASET_SIZE=10000
# ./train_4.sh ./training_${DATASET_SIZE}_question_prompt_document /workspace/data/ssm-retrieval/mamba2-8b/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model > _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_output.txt
# mv communication_output.txt _${current_datetime}_S_01_Q_A_training_${DATASET_SIZE}_communication_output.txt

