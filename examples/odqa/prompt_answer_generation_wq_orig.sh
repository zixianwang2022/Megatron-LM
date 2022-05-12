#!/bin/bash

pip install transformers==4.10.0 --use-feature=2020-resolver

# export CUDA_VISIBLE_DEVICES=0
# WORLD_SIZE=1
# DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
#                   --nnodes 1 \
#                   --node_rank 0 \
#                   --master_addr localhost \
#                   --master_port 0 \
#                   "

export CUDA_VISIBLE_DEVICES=$3
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port $((6000+$3)) \
                  "

CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m #(e.g., /357m)
VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)


###### The Webquestions

export EXP_NAME='wq_k10_357m'
INPUT_PATH_NEW=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/retrieval/predictions/dpr/webQuestion/WebQuestions-test.txt
PROMPT_PATH_NEW=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/retrieval/predictions/dpr/webQuestion/WebQuestions-train.txt 
ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/encoded_ctx_files_all_multisetdpr_queryctx_new.pickle
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_357m_gc_multisetdpr_new_rnd$2.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/generated_context_k10_357m_gc_multisetdpr_new_rnd$2.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/topk_context_k10_357m_gc_multisetdpr_new.json

# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_357m_gc_multisetdpr_new_top1.txt

# ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/encoded_ctx_files_all.pickle
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/train.json 

# ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/encoded_ctx_files_all_multisetdpr_queryctx.pickle
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/train.json
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-train.txt
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generation_train.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/generated_context_k10_357m_gc_multisetdpr_train.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/topk_context_k10_357m_gc_multisetdpr_train.json


### this is for megatron 530B gc + 357m ans
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/api_output_answer_generation_train_rnd1.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/api_generated_context_k10_530b_gc_multisetdpr_queryctx_p0.9_new_rnd1.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/topk_context_k10_530_gc_multisetdpr_train.json


### this is using the top-1 Context from the training DB
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generation_retrieve_nogolden_shift1.txt

### for topk context from retrieval + 357m ans
OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_top$2_ctx_357m_ans.txt

# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-test.txt

# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-train.txt
# # OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_357m_withnewnewGPTPrefix_l10_nogolden_withcontext_norandom_nq.txt 
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_357m_gc_nq.txt 
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/generated_context_k10_357m_nq.txt
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_357m_gc_tqa.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/generated_context_k10_357m_tqa.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/topk_from_tqa.json

############Ablation on WebQ
# ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/encoded_ctx_files_all_multisetdpr_queryctx.pickle
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/downloads/data/retriever/webq-train.json
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/downloads/data/retriever/webq-dev.json

# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/tqa_vs_dprwq.txt
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_357m_gc_dpr_random.txt
# # OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_357m_gc_dpr.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/generated_context_k10_357m_dpr.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/topk_from_dpr.json
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/tqa_vs_dprwq.json

# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/dprwq_vs_tqa.txt
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/dprwq_cgenfromtqa_with_samplesfromdprwq.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/generated_context_dprwq_vs_tqa.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/topk_context_dprwq_vs_tqa.json
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/topk_context_dprwq_cgenfromtqa_with_samplesfromdprwq.json

# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/retrieval_dprwq_k0.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/retrieval_dprwq.txt
# TOPK_CTX_PATH=

random_seed=$1
echo $random_seed

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 16 \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --load ${CHECKPOINT_PATH} \
        --fp16 \
        --DDP-impl torch \
        --tokenizer-type GPT2BPETokenizer \
        --input-file ${INPUT_PATH_NEW} \
        --output-file ${OUTPUT_PATH} \
        --prompt-file ${PROMPT_PATH_NEW} \
        --num-prompt-examples 10 \
        --out-seq-length 10 \
        --prompt-format 'ours' \
        --top-p-sampling 0.0 \
        --top-k-sampling 1 \
        --temperature 1.0 \
        --encoded-ctx-files ${ENCODED_CTX_FILE} \
        --task ODQA-CONTEXT-GEN-PROMPT \
        --with-context \
        --shift-steps 0 \
        --emb-type 'ctx' \
        --emb-type 'query_ctx' \
        --query-type 'question' \
        --random-seed $random_seed \
        --kth-context-from-retrieval $2 \

        # --save-context-path ${GEN_CTX_PATH} \
        # --save-topk-context-path ${TOPK_CTX_PATH} \
        # --is-context-generated \
        # --use-golden \
        # --shift-steps 0 \


        # --is-random \
        # --use-wiki-samples \
        # --api-prompt \
        # --megatron-api-url ${MEGATRON_API} \
        # --save-context-prompt-path ${CTX_PROMPT_PATH} \
        # --question-generation \

        
        # --emb-type 'ctx' \
        # --remove-duplicate-ctx \

        # --use-golden \
        # --is-random \


# CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-1.3b/ #(e.g., /357m)
# VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
# MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)

# export EXP_NAME='nq_k64_1.3b'
# # export ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/encoded_ctx_files_all_1.3b.pickle
# export ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/encoded_ctx_files_all_multisetdpr_queryctx.pickle

# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/train.json 
# # OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k0_1.3b_withnewnewGPTPrefix_l10_nogolden_withcontext_norandom_shift1.txt 
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_1.3b_gc_multisetdpr_queryctx_p0.9_rnd1.txt  
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/generated_context_k10_1.3b_gc_multisetdpr_queryctx_p0.9_rnd1.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/analysi_result/topk_context_k10_1.3b_multisetdpr_queryctx_p0.9_rnd1.json

# export EXP_NAME='tqa_k0_1.3b'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/test.json
# # INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/dev.json
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/train.json 
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k0_1.3b_withnewnewGPTPrefix_l10.txt 

# export EXP_NAME='wq_k1_357m'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-test.txt
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-train.txt 
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_1.3b_withnewnewGPTPrefix_l10.txt 


# export EXP_NAME='piqa_k1_357m'
# INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/PIQA/valid.jsonl #\  (the label file is valid-labels.lst)
# PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/PIQA/train.jsonl #\(the label file is train-labels.lst)
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/PIQA/output_answer_generations_k0_1.3b_l50_withgptneostyle.txt 


# python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
#         --num-layers 24 \
#         --hidden-size 2048 \
#         --num-attention-heads 32 \
#         --seq-length 2048 \
#         --max-position-embeddings 2048 \
#         --micro-batch-size 16 \
#         --vocab-file ${VOCAB_PATH} \
#         --merge-file ${MERGE_PATH} \
#         --load ${CHECKPOINT_PATH} \
#         --fp16 \
#         --DDP-impl torch \
#         --tokenizer-type GPT2BPETokenizer \
#         --input-file ${INPUT_PATH_NEW} \
#         --output-file ${OUTPUT_PATH} \
#         --prompt-file ${PROMPT_PATH} \
#         --num-prompt-examples 10 \
#         --out-seq-length 10 \
#         --prompt-format 'ours' \
#         --top-p-sampling 0.0 \
#         --top-k-sampling 1 \
#         --temperature 1.0 \
#         --encoded-ctx-files ${ENCODED_CTX_FILE} \
#         --task ODQA-CONTEXT-GEN-PROMPT \
#         --with-context \
#         --shift-steps 0 \
#         --use-golden \
#         --emb-type 'query_ctx' \
#         --is-context-generated \
#         --save-context-path ${GEN_CTX_PATH} \
#         --query-type 'question' \
#         --save-topk-context-path ${TOPK_CTX_PATH} \
#         --random-seed 4444 \
#         --question-generation \

        # --is-random \
        # --use-golden \


# NOTE: If you use api for the model generation, please use 
# the "--api-prompt" flag (setting this value as True). 
# 