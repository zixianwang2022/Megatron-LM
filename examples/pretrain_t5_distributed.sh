#!/bin/bash

set -xe

CORPUS="full-wikipedia"
SAVE_DIR="t5_main"
T5_CONFIG="base"
N_NODES=4
MODEL_PARALLEL=1

BASE_DIR="/lustre/fsw/adlr-nlp/dsachan/"
BASE_DATA_DIR="${BASE_DIR}/data"
DATA_DIR="${BASE_DATA_DIR}/${CORPUS}"
DATA_PATH="${DATA_DIR}/wikibooks_text_sentence"
VOCAB_FILE="${BASE_DIR}/bert_vocab/bert-large-uncased-vocab.txt"

BASE_CHKPT_DIR="${BASE_DIR}/checkpoints"
CHECKPOINT_PATH="${BASE_CHKPT_DIR}/${SAVE_DIR}_${CORPUS}_${T5_CONFIG}_mp${MODEL_PARALLEL}"

function config_base() {
    export CONFIG_ARGS="--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--kv-channels 64 \
--ffn-hidden-size 3072"
}

function config_large() {
    export CONFIG_ARGS="--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 16 \
--kv-channels 64 \
--ffn-hidden-size 4096"
}

function config_3B() {
    export CONFIG_ARGS="--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 32 \
--kv-channels 128 \
--ffn-hidden-size 16384"
}

function config_11B() {
    export CONFIG_ARGS="--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 128 \
--kv-channels 128 \
--ffn-hidden-size 65536"
}

if [ ${T5_CONFIG} == "base" ]; then
    config_base
elif [ ${T5_CONFIG} == "large" ]; then
    config_large
elif [ ${T5_CONFIG} == "3B" ]; then
    config_3B
elif [ ${T5_CONFIG} == "11B" ]; then
    config_11B
else
    echo "Invalid T5 model configuration"
    exit 1
fi

export OPTIONS=" \
         --model-parallel-size ${MODEL_PARALLEL} \
         --distributed-backend nccl \
         --num-layers 12 \
         --hidden-size 768 \
         --num-attention-heads 12 \
         --seq-length 512 \
         --decoder-seq-length 128 \
         --max-position-embeddings 512 \
         --data-path ${DATA_PATH} \
         --vocab-file ${VOCAB_FILE} \
         --fp16 \
         --train-iters 2000000 \
         --lr-decay-iters 990000 \
         --lr-decay-style linear \
         --batch-size 4 \
         --lr 0.0001 \
         --min-lr 0.00001 \
         --attention-dropout 0.1 \
         --weight-decay 1e-2 \
         --warmup 0.01 \
         --clip-grad 1.0 \
         --log-interval 100 \
         --save-interval 10000 \
         --eval-interval 1000 \
         --eval-iters 10 \
         --split 949,50,1 \
         --load ${CHECKPOINT_PATH} \
         --save ${CHECKPOINT_PATH} \
	       --checkpoint-activations \
	       --vocab-extra-ids 100 \
         --num-workers 2 "

sbatch --job-name="${SAVE_DIR}_${T5_CONFIG}_${CORPUS}_mp${MODEL_PARALLEL}" --nodes=${N_NODES} --export=OPTIONS,CONFIG_ARGS slurm_run_util.sh
