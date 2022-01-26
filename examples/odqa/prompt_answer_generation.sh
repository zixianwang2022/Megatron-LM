#!/bin/bash

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

# CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m #(e.g., /357m)
# VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
# MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k64.txt #\(e.g., /testseen_knowledge_generations.txt)

# python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
#         --num-layers 24 \
#         --hidden-size 1024 \
#         --num-attention-heads 16 \
#         --seq-length 2048 \
#         --max-position-embeddings 2048 \
#         --micro-batch-size 4 \
#         --vocab-file ${VOCAB_PATH} \
#         --merge-file ${MERGE_PATH} \
#         --load ${CHECKPOINT_PATH} \
#         --fp16 \
#         --DDP-impl torch \
#         --tokenizer-type GPT2BPETokenizer \
#         --input-file ${INPUT_PATH} \
#         --output-file ${OUTPUT_PATH} \
#         --prompt-file ${PROMPT_PATH} \
#         --num-prompt-examples 64 \
#         --task ODQA-PROMPT 


CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-1.3b/ #(e.g., /357m)
VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)

INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json #\  (e.g., /testseen_processed.txt)
PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/train.json #\(e.g., /testseen_knowledge_prompts.json)
OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k64_1.3b.txt #\(e.g., /testseen_knowledge_generations.txt)

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --num-layers 24 \
        --hidden-size 2048 \
        --num-attention-heads 32 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --load ${CHECKPOINT_PATH} \
        --fp16 \
        --DDP-impl torch \
        --tokenizer-type GPT2BPETokenizer \
        --input-file ${INPUT_PATH} \
        --output-file ${OUTPUT_PATH} \
        --prompt-file ${PROMPT_PATH} \
        --num-prompt-examples 64 \
        --task ODQA-PROMPT 

# NOTE: If you use api for the model generation, please use 
# the "--api-prompt" flag (setting this value as True). 
