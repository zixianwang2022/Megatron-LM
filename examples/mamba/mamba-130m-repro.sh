#!/bin/bash

#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --partition=large_runs_block1
#SBATCH --account=adlr_nlp_arch
#SBATCH --time=1-12:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --job-name=mamba-repro

# Notes for above:
# --nodes=1 for getting working on one node
# --account may be different for you
#   run `sacctmgr -nP show assoc where user=$(whoami) format=account`
# --ntasks-per-node=1 when using torchrun
# --partition=polar for short testing runs
# --partition=large_runs_block1 for long training runs
# --time=4:00:00 or less for short testing runs

# set -x

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

NAME="mamba-130m-repro"
IMAGE="/lustre/fsw/portfolios/adlr/users/bnorick/images/mamba-ssm.sqsh"
OUTPUT_ROOT="/lustre/fsw/portfolios/adlr/users/${USER}"

if [ -n "${SLURM_JOB_ID:-}" ] ; then
SCRIPT_DIR=$(dirname $(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}'))
else
SCRIPT_DIR=$(dirname $(realpath "$0"))
fi
EXAMPLES_DIR=$(dirname "${SCRIPT_DIR}")
MEGATRON_LM_DIR=$(dirname "${EXAMPLES_DIR}")

RUN_DIR="${OUTPUT_ROOT}/runs/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${RUN_DIR}/data-cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

DATA_PATH=/lustre/fsw/portfolios/adlr/users/bnorick/data/binidx/pile/gpt-neox-20b

SEQ_LEN=2048
TRAIN_SAMPLES=146484375  # 300B tokens / 2048
LR_WARMUP_SAMPLES=3906252  # 0.026666679773866397 * 146484375
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES-LR_WARMUP_SAMPLES))

# options notes:
# 32 x 8 = 256 global ("mini-") batch size
# --micro-batch-size 32 for single node, 8 GPUs total, no grad accum
# --micro-batch-size 8 for four nodes, 32 GPUs total, no grad accum
# If the global batch size cannot be computed in one iteration, then there will
# be more than one iteration, and the gradient will be accumulated over
# iterations before being applied.

options=" \
       --num-layers 24 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --train-samples ${TRAIN_SAMPLES} \
	--lr-warmup-samples ${LR_WARMUP_SAMPLES} \
	--lr-decay-samples ${LR_DECAY_SAMPLES} \
       --save ${CHECKPOINT_DIR} \
       --load ${CHECKPOINT_DIR} \
       --train-data-path ${DATA_PATH}/train.jsonl \
       --valid-data-path ${DATA_PATH}/val.jsonl \
       --data-cache-path ${DATACACHE_DIR} \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model "EleutherAI/gpt-neox-20b" \
       --distributed-backend nccl \
       --micro-batch-size 8 \
       --global-batch-size 256 \
       --lr 3.0e-3 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --bf16 \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_layer_spec \
       --tensorboard-dir ${TENSORBOARD_DIR}"

# We can use torchrun to launch eight processes on a single node using a single
# SLURM-allocated process on that node (set #SBATCH --ntasks-per-node=1).
# This is useful if we want to start an eight-GPU training job on a node, from
# an interactive session (a single SLURM-allocated process).
# run_cmd="torchrun --nproc_per_node 8 ${MEGATRON_LM_DIR}/pretrain_mamba.py ${options}"

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
