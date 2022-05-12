#!/bin/bash

pip install transformers==4.10.0 
# --use-feature=2020-resolver

export CUDA_VISIBLE_DEVICES=$3

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port $((6000+$3)) \
                  "

# export EXP_NAME='wq_1.3b_cgen_ans'
CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-1.3b/ #(e.g., /357m)
VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)

INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/retrieval/predictions/dpr/webQuestion/WebQuestions-test.txt
PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/retrieval/predictions/dpr/webQuestion/WebQuestions-train.txt 
ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/encoded_ctx_files_all_multisetdpr_queryctx_new.pickle

random_seed=$1
echo $random_seed

# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/output_answer_generations_k10_1.3b_gc_multisetdpr_queryctx_p0.9_$2.txt  # this means we fix all other parameters
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/1.3b/generated_context_k10_1.3b_gc_multisetdpr_queryctx_p0.9_$2.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/1.3b/topk_context_k10_1.3b_multisetdpr_queryctx_p0.9.json


### this is for megatron 530B gc + 1.3B ans
# export EXP_NAME='wq_530b_gc_1.3b_ans'
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/api_output_answer_generation_530b_gc_1.3b_ans_$2.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/api_generated_context_k10_530b_gc_multisetdpr_queryctx_p0.9_new_$2.txt
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/analysi_result/topk_context_k10_530_gc_multisetdpr_train.json


### this is for golden + 1.3B ans model
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/output_answer_generation_golden_1.3b_ans_$2.txt

### this is for top1 + 1.3B ans model
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/output_answer_generation_top1_ctx_1.3b_ans_$2.txt
OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/output_answer_generation_top1_ctx_1.3b_ans_$2_reversed.txt


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
        --random-seed $random_seed \
        # --use-golden \
        # --is-context-generated \
        # --save-context-path ${GEN_CTX_PATH} \
