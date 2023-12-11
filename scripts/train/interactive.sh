#!/bin/bash

set -u
unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

######## Arguments. ########

if [ "$#" != 2 ]; then
    echo "expected 2 args, found ${#}."
    exit 1
fi
USE_CORE=$1
ADD_RETRIEVER=$2
NPROCS=1 # 8
NWORKERS=32

REPO_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/retro-mcore-data"
RETRO_PROJECT_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/projects/wiki-tiny-core"

# CORPUS_ROOT="/lustre/fsw/portfolios/adlr/users/lmcafee/corpus-530b"
# DATA_PATH=" \
#   0.5 \
#   ${CORPUS_ROOT}/wiki-tiny-0/ds-0 \
#   0.5 \
#   ${CORPUS_ROOT}/wiki-tiny-1/ds-1 \
# "
# VOCAB_FILE=${ROOT_DIR}/retro/misc/vocab/gpt2-vocab.json
# MERGE_FILE=${ROOT_DIR}/retro/misc/vocab/gpt2-merges.txt
# TOKENIZER_ARGS=" \
#     --tokenizer-type GPT2BPETokenizer \
#     --vocab-file ${VOCAB_FILE} \
#     --merge-file ${MERGE_FILE} \
# "
# GLOBAL_BATCH_SIZE=256
# AUTO_ARGS=" \
#     ${TOKENIZER_ARGS} \
#     --seq-length 2048 \
#     --max-position-embeddings 2048 \
#     --global-batch-size ${GLOBAL_BATCH_SIZE} \
#     --eval-iters 100 \
#     --eval-interval 2000 \
#     --data-path ${DATA_PATH} \
#     --train-samples 100000  \
# "

NUM_LAYERS=12 # 4, [*12]
HIDDEN_SIZE=768 # 256, [512], *768
NUM_HEADS=16 # 12 # [4], 8, *12
MICRO_BATCH_SIZE=4 # [4], *8
LOG_INTERVAL=1 # 20
EXIT_INTERVAL=10

# --dataloader-type cyclic \
TP=1 # 8
ARGS=" \
    --exit-interval ${EXIT_INTERVAL} \
    \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --lr-decay-samples 99000 \
    --lr-warmup-samples 1000 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --log-interval ${LOG_INTERVAL} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.023 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --no-data-sharding \
"

if [ "$ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
else
    # --retro-no-verify-neighbor-count \
    # --retro-cyclic-train-iters 750000 \
    ARGS+=" \
      --retro-project-dir ${RETRO_PROJECT_DIR} \
      --retro-add-retriever \
      --num-workers ${NWORKERS} \
    "
    SCRIPT=pretrain_retro.py
fi

if [ "$USE_CORE" = "1" ]; then
    ARGS="${ARGS} --use-mcore-models"
fi

######## Command. ########

CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.
