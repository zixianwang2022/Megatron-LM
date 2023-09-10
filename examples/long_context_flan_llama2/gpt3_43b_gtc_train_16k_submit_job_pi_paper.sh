#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CKPT="gpt3-43b-multi-1.1t-gtc"
NAME="$CKPT-itp-16k-tian"

DIR=`pwd`
export SUBMIT_LOGS="$DIR/logs"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/gpt3-43b-multi-1.1t-gtc/tp8pp4/"
SAVE_DIR="$DIR/checkpoints/${NAME}"

if [[ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]]; then
    CHECKPOINT_DIR=${SAVE_DIR}
    opt=""
else
    opt="--no-load-rng \
    --no-load-optim \
    --finetune"
fi

mkdir -p $SAVE_DIR

TENSORBOARD_DIR="$DIR/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
# --clip-grad 1.0 \
. /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/data/tokens/multi-1.1t-gtc-blend-v0.1.sh

options="$opt \
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
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --num-layers 48 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --use-distributed-optimizer \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --train-iters 1000 \
    --lr 1.0e-5 \
    --min-lr 1.0e-5 \
    --lr-warmup-iters 20 \
    --log-interval 2 \
    --eval-interval 10 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --save-interval 200 \
    --save ${SAVE_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 99,1,0 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.010 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR}"

options=" \
    --seq-length 16384 \
    --max-position-embeddings 16384 \
    --rotary-seq-len-interpolation-factor 4 \
    --distributed-timeout-minutes 30 \
    --eval-iters 10 \
    $options"

# run_cmd="torchrun --nproc_per_node 8 ${DIR}/pretrain_gpt.py ${options}"
run_cmd="python -u ${DIR}/pretrain_gpt.py ${options}"

# export SUBMIT_ACCOUNT=llmservice_nlp_fm
LAUNCH="$ADLR_UTILS/mp_launch"
submit_job --gpu 8 --nodes 32 --email_mode never  --mounts "/lustre/fsw/adlr" --partition luna --image "gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:22.04-py3-eval" -c "$LAUNCH ${run_cmd}" -n "${NAME}" --duration 4 --exclude luna-0253,luna-0377  # --dependent_clones 3

# srun -l \
#      --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:23.04-py3-jbarker-revilm" \
#      --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr" \
#      --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"
# 
# set +x

