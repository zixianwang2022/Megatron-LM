#!/bin/bash

######## setup. ########

if [ "$#" != "2" ]; then
    echo "expected 2 args, found $#."
    exit 1
fi

set -u

# export CUDA_LAUNCH_BLOCKING=1 # llama
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

MODEL_SIZE=$2
if [ "${MODEL_SIZE}" = "7b" ]; then
    # {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}
    NPROCS=1
    ARGS=" \
        --hidden-size 4096 \
	--num-attention-heads 32 \
	--num-layers 32 \
	--norm-epsilon 1e-05 \
    "
elif [ "${MODEL_SIZE}" = "13b" ]; then
    # {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": -1}
    NPROCS=2
    ARGS=" \
        --hidden-size 5120 \
	--num-attention-heads 40 \
	--num-layers 40 \
	--norm-epsilon 1e-05 \
    "
elif [ "${MODEL_SIZE}" = "70b" ]; then
    # {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": -1}
    NPROCS=8
    ARGS=" \
        --hidden-size 8192 \
	--num-attention-heads 64 \
	--num-query-groups 8 \
	--num-layers 80 \
	--norm-epsilon 1e-05 \
    "
else
    echo "specialize for model size '${MODEL_SIZE}'."
    exit 1
fi

# >>>
MEGATRON_REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-llama2-loader"
LLAMA_REPO_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/llama"
TOKENIZER_PATH="/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/llama/tokenizer.model"

# MODEL_SIZE="7b"
# MODEL_SIZE="13b"
# MODEL_SIZE="70b"

COMMON_CHECKPOINT_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/llama/checkpoints"
MEGATRON_CHECKPOINT_DIR="${COMMON_CHECKPOINT_DIR}/megatron/${MODEL_SIZE}"
LLAMA_CHECKPOINT_DIR="${COMMON_CHECKPOINT_DIR}/llama/llama-2-${MODEL_SIZE}"
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
ARGS=" ${ARGS} \
    --no-masked-softmax-fusion \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
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
