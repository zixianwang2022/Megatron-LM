#!/bin/bash

#########################
# Evaluate the EM scores.
#########################
# pip install transformers==4.19.0 --use-feature=2020-resolver
pip install transformers==4.18.0 --use-feature=2020-resolver

export CUDA_VISIBLE_DEVICES=7

WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6007"


MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_rnd1.txt
GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json

MODEL_GEN_PATH_LIST=""
for i in `seq 1 8`
        do
        MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
        done
MODEL_GEN_PATH_LIST=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k64_multisetdpr_queryctx_530b_nocontext.txt


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

# python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
#         --num-layers 24 \
#         --hidden-size 1024 \
#         --num-attention-heads 16 \
#         --seq-length 2048 \
#         --max-position-embeddings 2048 \
#         --micro-batch-size 1 \
#         --task ODQA-EVAL-EM \
#         --guess-file ${MODEL_GEN_PATH} \
#         --answer-file ${GROUND_TRUTH_PATH} \