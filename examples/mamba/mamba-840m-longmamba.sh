#!/bin/bash

#SBATCH --ntasks-per-node=8
#SBATCH --nodes=16
#SBATCH --partition=polar,grizzly
#SBATCH --account=nvr_aialgo_ai4science
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --job-name=mamba-840m-longmamba-10k-lr3e-5-warmup20

# Notes for above:
# --nodes=1 for getting working on one node
# --account may be different for you
#   run `sacctmgr -nP show assoc where user=$(whoami) format=account`
# --ntasks-per-node=1 when using torchrun
# --partition=polar for short testing runs
# --partition=large_runs_block1 for long training runs
# --time=4:00:00 or less for short testing runs

# This script evolved from mamba-840m-comparison.sh, from Roger

set -x

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

NAME="mamba-840m-longmamba-10k-lr3e-5-warmup20"
# Roger's sqsh file extends Brandons (which has install of mamba-ssm)
# to include the installations needs for the eval tests
IMAGE="/lustre/fsw/portfolios/adlr/users/rwaleffe/images/mamba-ssm.sqsh"
# OUTPUT_ROOT="/lustre/fsw/portfolios/adlr/users/duncan"
OUTPUT_ROOT="/lustre/fsw/portfolios/nvr/users/wbyeon/dev/megatron-lm-mamba"

if [ -n "${SLURM_JOB_ID:-}" ] ; then
SCRIPT_DIR=$(dirname $(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}'))
else
SCRIPT_DIR=$(dirname $(realpath "$0"))
fi
EXAMPLES_DIR=$(dirname "${SCRIPT_DIR}")
MEGATRON_LM_DIR=$(dirname "${EXAMPLES_DIR}")

echo "> SCRIPT_DIR=${SCRIPT_DIR}"
echo "> EXAMPLES_DIR=${EXAMPLES_DIR}"
echo "> MEGATRON_LM_DIR=${MEGATRON_LM_DIR}"

#if [ -n "${SLURM_JOB_ID:-}" ] ; then
#  SCRIPT_DIR=$(dirname $(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}'))
#else
#  SCRIPT_DIR=$(dirname $(realpath "$0"))
#fi
#EXAMPLES_DIR=$(dirname "${SCRIPT_DIR}")
#MEGATRON_LM_DIR=$(dirname "${EXAMPLES_DIR}")
# MEGATRON_LM_DIR="/home/duncan/repos/megatron-lm"
#echo "SCRIPT_DIR: ${SCRIPT_DIR}"
#echo "MEGATRON_LM_DIR: ${MEGATRON_LM_DIR}"

RUN_DIR="${OUTPUT_ROOT}/runs/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${RUN_DIR}/data-cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

#DATA_PATH=/lustre/fsw/portfolios/adlr/users/bnorick/data/binidx/pile/gpt-neox-20b

# Get the data blend
#. multi-1.1t-gtc-blend-v0.1.sh
. ord_multi-1.1t-gtc-blend-v0.1.sh

# SEQ_LEN=4096
SEQ_LEN=16384

# TRAIN_SAMPLES=268554688  # 1.1T tokens / 4096
# 840m trained for 2,098,083 steps at batch size 128 = 268,554,624 samples
# TRAIN_SAMPLES=$((268554624+51200))  # LongMamba 128 (batch-size) x 400 (steps) = 51,200 extra steps
# LR_WARMUP_SAMPLES=2560  # LongMamba 20 (warm-up steps) x 128 (batch-size)
# LR_DECAY_SAMPLES=38640 # $((TRAIN_SAMPLES-LR_WARMUP_SAMPLES))

TRAIN_SAMPLES=269834624  # LongMamba 128 (batch-size) x 10000 (steps) + 268554624
# LR_WARMUP_SAMPLES=12800  # LongMamba 100 (warm-up steps) x 128 (batch-size)
LR_WARMUP_SAMPLES=2560  # LongMamba 20 (warm-up steps) x 128 (batch-size)
LR_DECAY_SAMPLES=1277440 #$((TRAIN_SAMPLES-LR_WARMUP_SAMPLES)) #1267200 # 

# TRAIN_SAMPLES=269834624 #268605824 #268557824
# Expect 838,860,800 extra tokens (41,200 train samples x 16,384 sequence length)

# Not sure if we should be using learning rate decay because it looks like
# LongMamba does not use it. I cannot see how to disable this from looking at
# the megatron code.

# options notes:
# 32 x 8 = 256 global ("mini-") batch size
# --micro-batch-size 32 for single node, 8 GPUs total, no grad accum
# --micro-batch-size 8 for four nodes, 32 GPUs total, no grad accum
# If the global batch size cannot be computed in one iteration, then there will
# be more than one iteration, and the gradient will be accumulated over
# iterations before being applied.

# Global batch size: 128 (to match LongMamba)
# Four machines = 32 GPUS
# So microbatch size is 128 / 32 = 4

# Checkpoints generated before Roger fixed position-embedding-type=none
# have embeddings, even though they're not needed for Mamba
# "learned_absolute" is actually the default for --position-embedding-type
#EMBEDDING_TYPE="learned_absolute"
EMBEDDING_TYPE="none"

# --rampup-batch-size 32 32 65324160 from nllvm1.1t
# Remember to re-enable checkpoint load and save
#       --load ${CHECKPOINT_DIR} \
#       --save ${CHECKPOINT_DIR} \
# Remember to set --eval-iters back to something greater than 1
# --no-load-optim \
options=" \
       --untie-embeddings-and-output-weights \
       --init-method-std 0.02 \
       --position-embedding-type ${EMBEDDING_TYPE} \
       --num-layers 48 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --train-samples ${TRAIN_SAMPLES} \
       --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
       --lr-decay-samples ${LR_DECAY_SAMPLES} \
       --save ${CHECKPOINT_DIR} \
       --load ${CHECKPOINT_DIR} \
       --no-load-optim \
       --data-path ${DATA_BLEND} \
       --data-cache-path ${DATACACHE_DIR} \
       --split 99,1,0 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --distributed-backend nccl \
       --micro-batch-size 1 \
       --global-batch-size 128 \
       --lr 3e-5 \
       --min-lr 3e-6 \
       --lr-decay-style cosine \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10 \
       --save-interval 10 \
       --eval-interval 10 \
       --eval-iters 32 \
       --bf16 \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_layer_spec \
       --tensorboard-dir ${TENSORBOARD_DIR}"

# We can use torchrun to launch eight processes on a single node using a single
# SLURM-allocated process on that node (set #SBATCH --ntasks-per-node=1).
# This is useful if we want to start an eight-GPU training job on a node, from
# an interactive session (a single SLURM-allocated process).
# run_cmd="torchrun --nproc_per_node 1 ${MEGATRON_LM_DIR}/pretrain_mamba.py ${options}"
# sh -c "${run_cmd}"; exit 0

# A job spread over multiple nodes and processes (all allocated via SLURM) will
# be properly coordinated using the following command (i.e. without using
# torchrun). I don't know if torchrun works with multi-GPU/multi-node.
run_cmd="python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${options}"

DATETIME=`date +'date_%Y-%m-%d_time_%H-%M-%S'`
# --container-mount-home doesn't seem to work. Explicit mapping added below.
srun -l \
     --container-image=${IMAGE} \
     --container-mounts=/lustre:/lustre,/home/${USER}:/home/${USER} \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${run_cmd}"

set +x
