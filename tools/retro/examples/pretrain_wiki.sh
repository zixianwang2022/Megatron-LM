#!/bin/bash

set -u
unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPROCS=8 # NPROCS must be <= number of GPUs.

######## Environment variables. ########
# (See get_cmd.sh for description of environment variables and $RETRO_ENV_VARS.)
. $RETRO_ENV_VARS

######## Data blend. ########
. ${BLEND_SCRIPT_DIR}/data_blend_${CORPUS}.sh
DATA_PATH=${DATA_BLEND}

######## Retro setup. ########
RETRO_WORKDIR=${RETRO_WORKDIRS}/${CORPUS}
RETRO_ADD_RETRIEVER=1
RETRO_CYCLIC_TRAIN_ITERS=750000
RETRO_NUM_NEIGHBORS=2

######## Arguments. ########
CHECKPOINT_DIR=${RETRO_WORKDIR}/checkpoints/interactive
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}
options=" \
    --load ${CHECKPOINT_DIR} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --train-samples  2037248  \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --log-interval 10 \
    --eval-iters 100 \
    --eval-interval 2000 \
    --data-path ${DATA_PATH} \
    --vocab-file ${GPT_VOCAB_FILE} \
    --merge-file ${GPT_MERGE_FILE} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.023 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --fp16 \
    --DDP-impl local \
    --dataloader-type cyclic \
    --no-data-sharding \
"

if [ "$RETRO_ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
else
    options="${options} \
    --retro-add-retriever \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-cyclic-train-iters ${RETRO_CYCLIC_TRAIN_ITERS} \
    --retro-num-neighbors ${RETRO_NUM_NEIGHBORS} \
    "
    SCRIPT=pretrain_retro.py
fi

python -m torch.distributed.launch \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
    ${SCRIPT} \
    ${options} \
