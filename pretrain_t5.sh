#!/bin/bash

set -xe

# The T5 model size. options: [base, large, 3B, 11B]
T5_CONFIG="base"

# Batch size per model instance
BATCH_SIZE_LOCAL=16

# Number of nodes to use
N_NODES=16

# Total batch size is calculated using:
# total-batch-size = n-nodes * 8 * batch-size-local / model-parallel-size

# Training data, vocab, and checkpoint directories 
DATA_DIR="/lustre/fsw/adlr-nlp/for_aws/data"
VOCAB_DIR="/lustre/fsw/adlr-nlp/for_aws/vocab"
CHECKPOINT_DIR="/lustre/fsw/adlr-nlp/for_aws/checkpoints"

# Training data.
DATA_PATH="${DATA_DIR}/wikibooks_text_sentence"

# Vocab file.
VOCAB_FILE="${VOCAB_DIR}/bert-large-uncased-vocab.txt"

# Directory to save checkpoints
CHECKPOINT_PATH="${CHECKPOINT_DIR}/t5_${T5_CONFIG}"


function config_base() {
    export CONFIG_ARGS="--num-layers 12 \
    	   --hidden-size 768 \
	   --num-attention-heads 12 \
	   --kv-channels 64 \
	   --ffn-hidden-size 3072 \
	   --model-parallel-size 1"
}

function config_large() {
    export CONFIG_ARGS="--num-layers 24 \
    	   --hidden-size 1024 \
	   --num-attention-heads 16 \
	   --kv-channels 64 \
	   --ffn-hidden-size 4096 \
	   --model-parallel-size 1"
}

function config_3B() {
    export CONFIG_ARGS="--num-layers 24 \
    	   --hidden-size 1024 \
	   --num-attention-heads 32 \
	   --kv-channels 128 \
	   --ffn-hidden-size 16384 \
	   --model-parallel-size 2"
}

function config_11B() {
    export CONFIG_ARGS="--num-layers 24 \
    	   --hidden-size 1024 \
	   --num-attention-heads 128 \
	   --kv-channels 128 \
	   --ffn-hidden-size 65536 \
	   --model-parallel-size 8"
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
       	 --bias-gelu-fusion \
	 --bias-dropout-fusion \
         --encoder-seq-length 512 \
         --decoder-seq-length 128 \
         --max-position-embeddings 512 \
         --batch-size ${BATCH_SIZE_LOCAL} \
         --checkpoint-activations \
         --train-iters 1000000 \
	 --lr 0.0001 \
	 --lr-decay-iters 1000000 \
	 --min-lr 0.00001 \
	 --lr-decay-style linear \
	 --warmup 0.01 \
	 --data-path ${DATA_PATH} \
	 --vocab-file ${VOCAB_FILE} \
	 --vocab-extra-ids 100 \
	 --split 949,50,1 \
	 --log-interval 100 \
	 --eval-interval 1000 \
	 --eval-iters 10 \
	 --save-interval 10000 \
	 --save ${CHECKPOINT_PATH} \
	 --load ${CHECKPOINT_PATH} \
	 --attention-dropout 0.1 \
	 --weight-decay 1e-2 \
	 --clip-grad 1.0 \
	 --fp16 \
	 --DDP-impl torch \
	 --num-workers 2 "

sbatch --job-name=T5_${T5_CONFIG} --nodes=${N_NODES} --export=OPTIONS,CONFIG_ARGS slurm_run_util.sh
