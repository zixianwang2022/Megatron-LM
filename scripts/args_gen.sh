#!/bin/bash

######## setup. ########

if [ "$#" != "1" ]; then
    echo "expected 1 arg, found $#."
    exit 1
fi

set -u

# export CUDA_LAUNCH_BLOCKING=1 # llama
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

# >>>
MEGATRON_REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-llama2-loader"
LLAMA_REPO_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/llama"
TOKENIZER_PATH="/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/llama/tokenizer.model"
MEGATRON_CHECKPOINT_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/llama/checkpoints/megatron/7b"
LLAMA_CHECKPOINT_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/llama/checkpoints/llama/llama-2-7b"
# +++
# MEGATRON_REPO_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/src/megatrons/megatron-lm-llama2-loader"
# LLAMA_REPO_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/llama/2/llama"
# # TOKENIZER_PATH="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
# TOKENIZER_PATH="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/llama/2/llama/tokenizer_hf/tokenizer.model"
# MEGATRON_CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/llama/2/llama/llama-2-7b-megatron"
# LLAMA_CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/llama/2/llama/llama-2-7b"
# <<<

######## args. ########

SCRIPT=scripts/generate.py
#     --attention-dropout 0.0 \
#     --hidden-dropout 0.0 \
#     --train-iters 100000 \
#     --lr-decay-iters 90000 \
#     --lr-warmup-iters 10000 \
#     --eval-interval 100 \
#     --eval-iters 0 \
#     --data-path /lustre/fsw/portfolios/adlr/users/lmcafee/retro/data/MTNLG/NIHExporter_shuf_text_document \
#     --split 99,1,0 \
#     --clip-grad 1.0 \
#     --weight-decay 0.1 \
#     --adam-beta1 0.9 \
#     --adam-beta2 0.95 \
#     --init-method-std 0.02 \
#     --log-params-norm \
#     --log-num-zeros-in-grad \
#     --use-flash-attn \
#     --apply-layernorm-1p \
#     --disable-bias-linear \
#     --rotary-percent 0.5 \
# --finetune \ ... doesn't load args from checkpoint
ARGS=" \
    --no-masked-softmax-fusion \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --norm-epsilon 1e-5 \
    --num-layers 32 \
    --micro-batch-size 1 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type Llama2 \
    --tokenizer-model ${TOKENIZER_PATH} \
    --load ${MEGATRON_CHECKPOINT_DIR} \
    --load-llama ${LLAMA_CHECKPOINT_DIR} \
    --no-load-optim \
    --no-load-rng \
    \
    --exit-duration-in-mins 230 \
    --fp16 \
    --DDP-impl local \
    --train-samples 1 \
    --min-lr 3.0e-5 \
    --lr 3.0e-4 \
    --lr-decay-style cosine \
    \
    --untie-embeddings-and-output-weights \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --swiglu-llama \
    \
    --gen-model $1 \
    --norm-type rms \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
    --no-query-key-layer-scaling \
"

# eof.
