#!/bin/bash

set -u
unset NCCL_DEBUG
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# . ./get_vars.sh

. ${BLEND_SCRIPT_DIR}/gpt3_blend_wiki.sh
DATA_PATH=${DATA_BLEND}

RETRO_ADD_RETRIEVER=1
RETRO_WORKDIR=${RETRO_WORKDIRS}/wiki
RETRO_CYCLIC_TRAIN_ITERS=750000
RETRO_NNBRS=2 # *2, 10

NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_HEADS=12
MICRO_BATCH_SIZE=4

CHECKPOINT_DIR=${RETRO_WORKDIR}/checkpoints/interactive
# CHECKPOINT_DIR=${RETRO_WORKDIR}/checkpoints/inference
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}

#     --save-interval 10000 \
#     --tensorboard-dir ${TENSORBOARD_DIR} \
#     --log-validation-ppl-to-tensorboard \
options=" \
    --load ${CHECKPOINT_DIR} \
    --no-async-tensor-model-parallel-allreduce \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
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

if [ "$ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
else
    options="${options} \
    --retro-add-retriever \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-cyclic-train-iters ${RETRO_CYCLIC_TRAIN_ITERS} \
    --retro-nnbrs ${RETRO_NNBRS} \
    "
    SCRIPT=pretrain_gpt_retro.py
fi

# NPROCS=1
NPROCS=16
python -m torch.distributed.launch \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
    ${SCRIPT} \
    ${options} \
