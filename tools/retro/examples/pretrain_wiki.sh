#!/bin/bash

set -u

. /gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/misc/gpt3_blend_wiki.sh
DATA_PATH=${DATA_BLEND}

# BPE_DIR="/lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/bpe"
# VOCAB_FILE=${BPE_DIR}/gpt2-vocab.json \
# MERGE_FILE=${BPE_DIR}/gpt2-merges.txt \
VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json
MERGE_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
NUM_LAYERS=12 # 4, [*12]
HIDDEN_SIZE=768 # 256, [512], *768
NUM_HEADS=12 # [4], 8, *12
MICRO_BATCH_SIZE=4 # 2[k=10], 4[draco-rno], *8
ADD_RETRIEVER=1

# >>>
CHECKPOINT_DIR=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/checkpoints/interactive
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}
# <<<

#     --tensorboard-dir ${TENSORBOARD_DIR} \
#     --log-validation-ppl-to-tensorboard \
#     --loss-scale 1024 \
options=" \

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
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --save-interval 10000 \
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
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NUM_LAYERS=12 # 4, *12
# HIDDEN_SIZE=512 # 256, 512, *768
# NUM_HEADS=4 # 4, 8, *12
# # RETRO_CYCLIC_TRAIN_ITERS=750000
# RETRO_CYCLIC_TRAIN_ITERS=1
# MICRO_BATCH_SIZE=4 # [4], *8
# GLOBAL_BATCH_SIZE=4 # [4], *256
# EVAL_ITERS=1 # [0], *100
# ADD_RETRIEVER=1
# options=" \
#     --tensor-model-parallel-size 1 \
#     --pipeline-model-parallel-size 1 \
#     --num-layers ${NUM_LAYERS} \
#     --hidden-size ${HIDDEN_SIZE} \
#     --num-attention-heads ${NUM_HEADS} \
#     --seq-length 2048 \
#     --max-position-embeddings 2048 \
#     --micro-batch-size ${MICRO_BATCH_SIZE} \
#     --global-batch-size ${GLOBAL_BATCH_SIZE} \
#     --train-samples  2037248  \
#     --lr-decay-samples 166400000 \
#     --lr-warmup-samples 162761 \
#     --lr 6.0e-4 \
#     --min-lr 6.0e-5 \
#     --lr-decay-style cosine \
#     --log-interval 1 \
#     --eval-iters ${EVAL_ITERS} \
#     --eval-interval 2000 \
#     --data-path ${DATA_PATH} \
#     --vocab-file ${VOCAB_FILE} \
#     --merge-file ${MERGE_FILE} \
#     --save-interval 10000 \
#     --split 98,2,0 \
#     --clip-grad 1.0 \
#     --weight-decay 0.1 \
#     --adam-beta1 0.9 \
#     --adam-beta2 0.95 \
#     --init-method-std 0.023 \
#     --log-params-norm \
#     --log-num-zeros-in-grad \
#     --fp16 \
#     --loss-scale 1024 \
#     --DDP-impl local \
#     --dataloader-type cyclic \
#     --no-data-sharding \
# "
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if [ "$ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
else
    RETRO_WORKDIR=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki
    # RETRO_CYCLIC_TRAIN_ITERS=750000
    RETRO_CYCLIC_TRAIN_ITERS=100 # 1, 20
    options="${options} \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    --retro-cyclic-train-iters ${RETRO_CYCLIC_TRAIN_ITERS} \
    "
    SCRIPT=pretrain_gpt_retro.py
fi

unset NCCL_DEBUG

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
