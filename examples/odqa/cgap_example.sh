#!/bin/bash

# this gives you an concrete example
pip install transformers==4.10.0 --use-feature=2020-resolver

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

export ENCODED_CTX_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/encoded_ctx_files_all_multisetdpr_queryctx.pickle
INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json
PROMPT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/train.json

OUTPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_357m_ans.txt 
GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9.txt
TOPK_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/analysi_result/topk_context_k10_357m_multisetdpr_queryctx_p0.9.json


python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
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
        --random-seed $1 \
        --use-golden \
        --save-context-path ${GEN_CTX_PATH} \
        --is-context-generated \
        --save-topk-context-path ${TOPK_CTX_PATH} \