#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

NAME="flamingo-2b-pretrain-1e-4-ocr-randominit-aligned-train-samples-new-withdocidx-mr-wds"
CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/${NAME}"

dataset="GPT"
samples=1000
task="OCR"

EVAL_PATH="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/data/OCR_valid"
resolution=1024
VISUAL_ARCH="SAM_L"
VISUAL_TYPE="sam"
VISUAL_DIR="${CHECKPOINT_DIR}/${VISUAL_TYPE}"

iter=50000

CUDA_VISIBLE_DEVICES=0 MASTER_PORT=44140 python generation/generate_samples_flamingo_nonparallel.py \
       --use-flash-attn \
       --apply-layernorm-1p \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --no-position-embedding \
       --use-rotary-position-embeddings \
       --rotary-percent 0.5 \
       --swiglu \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --exit-duration-in-mins 230 \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --max-position-embeddings 4096 \
       --no-masked-softmax-fusion \
       --load ${CHECKPOINT_DIR} \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --bf16 \
       --micro-batch-size 1 \
       --seq-length 96 \
       --out-seq-length 300 \
       --temperature 1.0 \
       --dataset $dataset \
       --visual-arch ${VISUAL_ARCH} \
       --visual-path ${VISUAL_DIR} \
       --visual-type ${VISUAL_TYPE} \
       --num-samples $samples \
       --add-gated-xattn \
       --img-h $resolution \
       --img-w $resolution \
       --seed 153 \
       --top_k 1 \
       --task $task \
       --perceiver-type none \
       --eval-path $EVAL_PATH \
       --load-iter ${iter} \
       --genfile ./generated_files/$NAME-$iter-$dataset-$task-${resolution}px.jsonl \
       --align-to-old \
       # --with-space \
       # --fp32SAM \
