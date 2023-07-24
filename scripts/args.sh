#!/bin/bash

######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-llama2-loader"

######## checkpoint. ########

# CHECKPOINT_DIR="/lustre/fs1/portfolios/adlr/users/lmcafee/main-dev/checkpoints"
# TENSORBOARD_DIR="$CHECKPOINT_DIR/tensorboard"
# mkdir -p ${TENSORBOARD_DIR}

######## args. ########

SCRIPT=pretrain_gpt.py
#     --use-distributed-optimizer \
#     --fp16 --loss-scale 8192 \
#     --train-samples 10000000 \
#     --lr-decay-samples 9000000 \
#     --lr-warmup-samples 1000000 \
# ARGS=" \
#     --save-interval 10000 \
#     --save ${CHECKPOINT_DIR} \
#     --load ${CHECKPOINT_DIR} \
#     --tensorboard-dir ${TENSORBOARD_DIR} \
#     \
SEQ_LENGTH=1024
# SEQ_LENGTH=2048
ARGS=" \
    --log-interval 1 \
    --exit-interval 100000 \
    --num-workers 0 \
    \
    --num-layers 8 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --train-iters 100000 \
    --lr-decay-iters 90000 \
    --lr-warmup-iters 10000 \
    --lr 3.0e-4 \
    --min-lr 3.0e-5 \
    --lr-decay-style cosine \
    --eval-interval 100 \
    --eval-iters 0 \
    --data-path /lustre/fs1/portfolios/adlr/users/lmcafee/retro/data/MTNLG/NIHExporter_shuf_text_document \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --swiglu \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fs1/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
"

# eof.
