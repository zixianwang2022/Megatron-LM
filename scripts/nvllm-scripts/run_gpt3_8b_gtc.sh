#!/bin/bash

#SBATCH -p luna -A adlr -t 4:00:00 --nodes=32 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=adlr-nlp-largelm:gpt3-8b-multi-1.1t-gtc

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

NAME="gpt3-8b-multi-1.1t-gtc"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/${NAME}"

TENSORBOARD_DIR="$DIR/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/data/tokens/multi-1.1t-gtc-blend-v0.1.sh

options=" \
    --sequence-parallel \
    --recompute-activations \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --rampup-batch-size 64 64 65324160 \
    --train-samples 268554688 \
    --lr-decay-samples 255126953 \
    --lr-warmup-samples 122071 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --save-interval 10000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.010 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR}"

run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_gpt.py ${options}"

srun -l \
     --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:22.12-py3-eval" \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

