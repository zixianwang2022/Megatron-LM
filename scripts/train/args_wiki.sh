#!/bin/bash

# RETRO_PROJECT_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/projects/wiki-tiny-core"
# RETRO_PROJECT_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/projects/wiki-core"
RETRO_PROJECT_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/projects/wiki-core-bert-fast"

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# NUM_LAYERS=12 # 4, [*12]
# HIDDEN_SIZE=768 # 256, [512], *768
# NUM_HEADS=16 # 12 # [4], 8, *12
# MICRO_BATCH_SIZE=4 # [4], *8
# # LOG_INTERVAL=1 EXIT_INTERVAL=10
# LOG_INTERVAL=10 EXIT_INTERVAL=100

# # --dataloader-type cyclic \
# # --split 98,2,0 \
# TP=1 # 8
# ARGS=" \
#     --exit-interval ${EXIT_INTERVAL} \
#     \
#     --tensor-model-parallel-size ${TP} \
#     --pipeline-model-parallel-size 1 \
#     --num-layers ${NUM_LAYERS} \
#     --hidden-size ${HIDDEN_SIZE} \
#     --num-attention-heads ${NUM_HEADS} \
#     --micro-batch-size ${MICRO_BATCH_SIZE} \
#     --lr-decay-samples 99000 \
#     --lr-warmup-samples 1000 \
#     --lr 6.0e-4 \
#     --min-lr 6.0e-5 \
#     --lr-decay-style cosine \
#     --log-interval ${LOG_INTERVAL} \
#     --split 99,1,0 \
#     --clip-grad 1.0 \
#     --weight-decay 0.1 \
#     --adam-beta1 0.9 \
#     --adam-beta2 0.95 \
#     --init-method-std 0.023 \
#     --log-params-norm \
#     --log-num-zeros-in-grad \
#     --bf16 \
#     --no-data-sharding \
# "
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --loss-scale 1024 \
NUM_LAYERS=12 # 4, [*12]
HIDDEN_SIZE=768 # 256, [512], *768
NUM_HEADS=12 # [4], 8, *12
MICRO_BATCH_SIZE=4 # [4], *8

# --DDP-impl local \
# --dataloader-type cyclic \
# --seq-length 2048 \
# --max-position-embeddings 2048 \
# --global-batch-size 256 \
# --train-samples  2037248  \
# --eval-iters 100 \
# --eval-interval 2000 \
# --data-path ${DATA_PATH} \
# --vocab-file ${VOCAB_FILE} \
# --merge-file ${MERGE_FILE} \
 
ARGS=" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
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

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>
# if [ "$ADD_RETRIEVER" = "0" ]; then
#     SCRIPT=pretrain_gpt.py
# else
#     # --retro-no-verify-neighbor-count \
#     # --retro-cyclic-train-iters 750000 \
#     ARGS+=" \
#       --retro-project-dir ${RETRO_PROJECT_DIR} \
#       --retro-add-retriever \
#       --num-workers ${NWORKERS} \
#     "
#     SCRIPT=pretrain_retro.py
# fi
# +++
SCRIPT="pretrain_retro.py"
ARGS+=" --retro-project-dir ${RETRO_PROJECT_DIR}"
ARGS+=" --dataloader-type cyclic"
ARGS+=" --retro-cyclic-train-iters 750000"
# ARGS+=" --retro-cyclic-train-iters 2037248"
ARGS+=" --num-workers ${NWORKERS}"
# ARGS+=" --num-workers 32"
# ARGS+=" --num-workers 1"

if [ "$ADD_RETRIEVER" = "1" ]; then
    ARGS+=" --retro-add-retriever"
fi
# <<<

if [ "$USE_CORE" = "1" ]; then
    ARGS="${ARGS} --use-mcore-models"
fi

# eof
