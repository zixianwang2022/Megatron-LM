#!/bin/bash

#########################
# Evaluate the EM scores.
#########################
export CUDA_VISIBLE_DEVICES=4

WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6004"

# NQ 
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_withnewnewGPTPrefix_l10_usegolden_withcontext_norandom_multisetdpr_qqctx.txt 
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9.txt # (e.g., /testseen_knowledge_generations.txt)
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/NQ/openai_babbage_ours_np_0.txt # (e.g., /testseen_knowledge_generations.txt)
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/NQ/openai_babbage_GPT-3_np_0.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/NQ/openai_babbage_Eleuther-AI_np_0.txt
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json #\ (e.g., /testseen_knowledge_reference.txt)
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9.txt

# # TQA test set
MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_357m_withnewnewGPTPrefix_l10_usegolden_withcontext_norandom_multisetdpr_qq.txt # (e.g., /testseen_knowledge_generations.txt)
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all.txt # (e.g., /testseen_knowledge_generations.txt)
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/TQA/openai_babbage_ours_np_0.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/TQA/openai_babbage_GPT-3_np_0.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/TQA/openai_babbage_Eleuther-AI_np_0.txt
GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/test.json #\ (e.g., /testseen_knowledge_reference.txt)
GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/generated_context_k10_357m_multisetdpr_queryctx_p0.9_all.txt

# TQA dev set
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k64_1.3b_dev.txt # (e.g., /testseen_knowledge_generations.txt)
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/dev.json #\ (e.g., /testseen_knowledge_reference.txt)

# WebQuestions test set
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_357m_gc_nq.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/WQ/openai_babbage_ours_np_64.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/WQ/openai_ada_GPT-3_np_64.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/WQ/openai_babbage_Eleuther-AI_np_1.txt
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-test.txt #\ (e.g., /testseen_knowledge_reference.txt)


# PIQA
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/PIQA/output_answer_generations_k0_357m_l50_withgptneostyle_p0.0k1t1.0_new.txt # (e.g., /testseen_knowledge_generations.txt)
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/PIQA/valid-labels.lst #\ (e.g., /testseen_knowledge_reference.txt)


python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 1 \
        --task ODQA-EVAL-EM \
        --guess-file ${MODEL_GEN_PATH} \
        --answer-file ${GROUND_TRUTH_PATH} \
        --save-context-path ${GEN_CTX_PATH} \


############################################
# Evaluate BLEU, METEOR, and ROUGE-L scores.
############################################

# We follow the nlg-eval (https://github.com/Maluuba/nlg-eval) to 
# evaluate the BLEU, METEOR, and ROUGE-L scores. 

# To evaluate on these metrics, please setup the environments based on 
# the nlg-eval github, and run the corresponding evaluation commands.

# nlg-eval \
#     --hypothesis=<PATH_OF_THE_KNOWLEDGE_GENERATION> \
#     --references=<PATH_OF_THE_GROUND_TRUTH_KNOWLEDGE>
