#!/bin/bash

pip install transformers==4.10.0 
# --use-feature=2020-resolver

export CUDA_VISIBLE_DEVICES=$3,$((1+$3)),$((2+$3)),$((3+$3))

echo $CUDA_VISIBLE_DEVICES

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port $((6000+$3)) \
                  "

CHECKPOINT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-8.3b/ #(e.g., /357m)
VOCAB_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json #(e.g., /gpt2-vocab.json)
MERGE_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt #(e.g., /gpt2-merges.txt)

export EXP_NAME='nq_8.3b_cgen_ans'

export ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/encoded_ctx_files_all_multisetdpr_queryctx.pickle
INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json
PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/train.json
# INPUT_PATH_NEW=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/retrieval/predictions/dpr/nq/test.json

random_seed=$1
echo $random_seed

### 8.3b gc + 8.3b ans

OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/8.3b/output_answer_generations_k10_8.3b_gc_multisetdpr_queryctx_p0.9_$2.txt  # this means we fix all other parameters
GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/8.3b/generated_context_k10_8.3b_gc_multisetdpr_queryctx_p0.9_$2.txt

###  530B gc + 1.3B ans
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/api_output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_$2.txt  # this means we fix all other parameters
# TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/analysi_result/1.3b/topk_context_k10_1.3b_multisetdpr_queryctx_p0.9.json
# # GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/api/api_generated_context_k10_530b_gc_multisetdpr_queryctx_p0.9_$2_new.txt
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/api/api_generated_context_k10_530b_gc_multisetdpr_queryctx_p0.9_$2.txt


### golden + 1.3B ans
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/output_answer_generations_k10_golden_ctx_multisetdpr_queryctx_p0.9_$2.txt  # this means we fix all other parameters

#### top1 C1 + 1.3B ans
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_$2.txt  # this means we fix all other parameters
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_$2_reversed.txt  # this means we fix all other parameters


#### retrieval: top-k as context + 1.3B ans model
# OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/output_answer_generations_k10_top$2_ctx_1.3b_ans.txt 

# --pipeline-model-parallel-size 1 \
python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --tensor-model-parallel-size 4 \
        --num-layers 40 \
        --hidden-size 4096 \
        --num-attention-heads 64 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 1 \
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
        --emb-type 'query_ctx' \
        --query-type 'question' \
        --random-seed $random_seed \
        --use-golden \
        --save-context-path ${GEN_CTX_PATH} \
        --is-context-generated \

        # --kth-context-from-retrieval $2 \
