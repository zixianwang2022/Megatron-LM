#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6001"


CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m #(e.g., /357m)
VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)

# export EXP_NAME='nq_k0_357m'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json #\  (e.g., /testseen_processed.txt)
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/train.json #\(e.g., /testseen_knowledge_prompts.json)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k0_357m_withnewnewGPTPrefix_l10.txt #\(e.g., /testseen_knowledge_generations.txt)

# export EXP_NAME='tqa_k1_357m'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/test.json #\  (e.g., /testseen_processed.txt)
# # INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/dev.json #\  (e.g., /testseen_processed.txt)
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/train.json #\(e.g., /testseen_knowledge_prompts.json)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k0_357m_withnewnewGPTPrefix_l10.txt #\(e.g., /testseen_knowledge_generations.txt)

# export EXP_NAME='wq_k1_357m'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-test.txt #\  (e.g., /testseen_processed.txt)
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-train.txt #\(e.g., /testseen_knowledge_prompts.json)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_357m_withnewnewGPTPrefix_l10.txt #\(e.g., /testseen_knowledge_generations.txt)


# PIQA dataset
export EXP_NAME='k0_357m_l50_withgptneostyle_p0.5k0t1.0'
INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/PIQA/valid.jsonl #\  (the label file is valid-labels.lst)
PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/PIQA/train.jsonl #\(the label file is train-labels.lst)
OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/PIQA/output_answer_generations_${EXP_NAME}.txt #\(e.g., /testseen_knowledge_generations.txt)


python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 128 \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --load ${CHECKPOINT_PATH} \
        --fp16 \
        --DDP-impl torch \
        --tokenizer-type GPT2BPETokenizer \
        --input-file ${INPUT_PATH} \
        --output-file ${OUTPUT_PATH} \
        --prompt-file ${PROMPT_PATH} \
        --num-prompt-examples 0 \
        --out-seq-length 50 \
        --exp-name ${EXP_NAME} \
        --top-p-sampling 0.5 \
        --top-k-sampling 0 \
        --temperature 1.0 \
        --task ODQA-PROMPT


# CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-1.3b/ #(e.g., /357m)
# VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
# MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)

# export EXP_NAME='nq_k64_1.3b'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json #\  (e.g., /testseen_processed.txt)
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/train.json #\(e.g., /testseen_knowledge_prompts.json)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k0_1.3b_withnewnewGPTPrefix_l10.txt #\(e.g., /testseen_knowledge_generations.txt)

# export EXP_NAME='tqa_k0_1.3b'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/test.json #\  (e.g., /testseen_processed.txt)
# # INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/dev.json #\  (e.g., /testseen_processed.txt)
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/train.json #\(e.g., /testseen_knowledge_prompts.json)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k0_1.3b_withnewnewGPTPrefix_l10.txt #\(e.g., /testseen_knowledge_generations.txt)

# export EXP_NAME='wq_k1_357m'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-test.txt #\  (e.g., /testseen_processed.txt)
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-train.txt #\(e.g., /testseen_knowledge_prompts.json)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_1.3b_withnewnewGPTPrefix_l10.txt #\(e.g., /testseen_knowledge_generations.txt)


# export EXP_NAME='piqa_k1_357m'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/PIQA/valid.jsonl #\  (the label file is valid-labels.lst)
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/PIQA/train.jsonl #\(the label file is train-labels.lst)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/PIQA/output_answer_generations_k0_1.3b_l50_withgptneostyle.txt #\(e.g., /testseen_knowledge_generations.txt)


# python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
#         --num-layers 24 \
#         --hidden-size 2048 \
#         --num-attention-heads 32 \
#         --seq-length 2048 \
#         --max-position-embeddings 2048 \
#         --micro-batch-size 128 \
#         --vocab-file ${VOCAB_PATH} \
#         --merge-file ${MERGE_PATH} \
#         --load ${CHECKPOINT_PATH} \
#         --fp16 \
#         --DDP-impl torch \
#         --tokenizer-type GPT2BPETokenizer \
#         --input-file ${INPUT_PATH} \
#         --output-file ${OUTPUT_PATH} \
#         --prompt-file ${PROMPT_PATH} \
#         --num-prompt-examples 0 \
#         --out-seq-length 50 \
#         --exp-name ${EXP_NAME} \
#         --task ODQA-PROMPT 

# NOTE: If you use api for the model generation, please use 
# the "--api-prompt" flag (setting this value as True). 
