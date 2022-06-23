#!/bin/bash

#########################
# Evaluate the EM scores.
#########################
# pip install transformers==4.19.0 --use-feature=2020-resolver
pip install transformers==4.18.0 --use-feature=2020-resolver


export CUDA_VISIBLE_DEVICES=0

WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6007"

MODEL_GEN_PATH=<MODEL_GEN_PATH>
GROUND_TRUTH_PATH=<YOUR_INPUT_FILE_PATH>
MODEL_GEN_PATH_LIST=<MULTI_MODEL_GEN_PATH> 

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 1 \
        --task ODQA-EVAL-EM \
        --guess-file ${MODEL_GEN_PATH_LIST} \
        --answer-file ${GROUND_TRUTH_PATH} \