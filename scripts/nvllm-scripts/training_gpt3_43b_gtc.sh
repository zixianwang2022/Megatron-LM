#!/bin/bash

#SBATCH -p large_runs --reservation adlr_388 -A adlr -t 4:00:00 --nodes=384 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --job-name=adlr-nlp-largelm:gpt3-43b-multi-1.1t-gtc --exclude=luna-[0471,0538,0008,0151,0390]

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_DEBUG=INFO

NAME="gpt3-43b-multi-1.1t-gtc"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

#CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/mpatwary/checkpoints/gpt3/gpt3-next-llm-multi-1.1t-gtc/${NAME}"
CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/${NAME}"
BLENDABLE_INDEX="/lustre/fsw/adlr/adlr-gtc-43b/data/blendable-index/${NAME}"

TENSORBOARD_DIR="$DIR/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/data/tokens/multi-1.1t-gtc-blend-v0.1.sh

options=" \
    --unsafe-flag-for-gpt-speedup \
    --use-container-fused-kernels \
    --blendable-index-path ${BLENDABLE_INDEX} \
    --sequence-parallel \
    --recompute-activations \
    --use-flash-attn \
    --overlap-p2p-communication \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 220 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --num-layers-per-virtual-pipeline-stage 1 \
    --num-layers 48 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 768 \
    --rampup-batch-size 192 192 65323968 \
    --train-samples 268554688 \
    --lr-decay-samples 255126953 \
    --lr-warmup-samples 179037 \
    --lr 9.0e-5 \
    --min-lr 9.0e-6 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --save-interval 2000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.007 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR}"

run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_gpt.py ${options}"

     #--container-image "gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:22.12-py3-eval" \
     #--container-image "/lustre/fsw/adlr/adlr-nlp/images/adlr+megatron-lm+pytorch+22.12-py3-eval.sqsh" \

srun -l \
     --container-image "/lustre/fsw/adlr/adlr-nlp/images/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels.sqsh" \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

