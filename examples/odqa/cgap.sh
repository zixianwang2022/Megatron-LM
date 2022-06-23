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

CHECKPOINT_PATH=<YOUR_MODEL_PATH>
VOCAB_PATH=<YOUR_VOCAB_PATH>
MERGE_PATH=<YOUR_MERGE_PATH>

ENCODED_CTX_FILE=<ENCODED_CTX_FILE_SAVE_PATH> \
INPUT_PATH=<YOUR_INPUT_FILE_PATH>
PROMPT_PATH=<YOUR_PROMPT_FILE_PATH>


OUTPUT_PATH=<YOUR_ANSWER_PREDICTION_SAVE_PATH>
GEN_CTX_PATH=<YOUR_CGEN_SAVED_PATH>
TOPK_CTX_PATH=<YOUR_TOPK_LIST_SAVED_PATH>


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