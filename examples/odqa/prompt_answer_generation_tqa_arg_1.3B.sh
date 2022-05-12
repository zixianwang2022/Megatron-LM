#!/bin/bash

pip install transformers==4.10.0 --use-feature=2020-resolver

export CUDA_VISIBLE_DEVICES=$3

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port $((6000+$3)) \
                  "

CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-1.3b/ #(e.g., /357m)
VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)


export EXP_NAME='tqa_1.3b_cgen'
# INPUT_PATH_NEW=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/retrieval/predictions/dpr/triviaQA/test.json
# # # # export ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/encoded_ctx_files_all_multisetdpr_queryctx_traingoodk0.pickle
export ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/encoded_ctx_files_all_multisetdpr_queryctx.pickle
INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/test.json
PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/train.json
# # # # # PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/train_good_examples.json
# # OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_357m_withnewnewGPTPrefix_l10_nogolden_withcontext_norandom_multisetdpr_qc.txt
# # OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_11313_rmduplicatectx.txt  
# # GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/generated_context_k10_357m_multisetdpr_queryctx_p0.9_11313_rmduplicatectx.txt
# # TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/analysi_result/topk_context_k10_357m_multisetdpr_queryctx_p0.9_11313_rmduplicatectx.json
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/qg/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_fewshot.txt  
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/qg/generated_context_k10_357m_multisetdpr_queryctx_p0.9_fewshot.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/analysi_result/qg/topk_context_k10_357m_multisetdpr_queryctx_p0.9_fewshot.json

### TQA Test Data

# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/output_answer_generations_k10_1.3b_gc_multisetdpr_queryctx_p0.9_$2.txt  
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/1.3b/generated_context_k10_1.3b_multisetdpr_queryctx_p0.9_$2.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/analysi_result/1.3b/topk_context_k10_1.3b_multisetdpr_queryctx_p0.9.json


### TQA topk + 1.3B ans model
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_$2.txt 
OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_$2_reversed.txt 

### golden + 1.3B ans
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/output_answer_generations_k10_golden_ctx_multisetdpr_queryctx_p0.9_$2.txt


python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --num-layers 24 \
        --hidden-size 2048 \
        --num-attention-heads 32 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 12 \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --load ${CHECKPOINT_PATH} \
        --fp16 \
        --DDP-impl torch \
        --tokenizer-type GPT2BPETokenizer \
        --input-file ${INPUT_PATH} \
        --output-file ${OUTPUT_PATH} \
        --prompt-file ${PROMPT_PATH} \
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
        --random-seed $1 \
        # --use-golden \
        # --is-context-generated \
        # --save-context-path ${GEN_CTX_PATH} \
        # --save-topk-context-path ${TOPK_CTX_PATH} \



        # --question-generation \

        
        # --emb-type 'ctx' \
        # --remove-duplicate-ctx \

        # --use-golden \
        # --is-random \

