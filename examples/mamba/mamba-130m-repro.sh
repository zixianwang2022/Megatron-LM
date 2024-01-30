#!/bin/bash

#SBATCH -p batch_block1,batch_block2,batch_block3,batch_block4 -A llmservice_nlp_fm -t 4:00:00 --nodes=4 --exclusive --gres=gpu:8 --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=llmservice_nlp_fm-adlr-nlp-largelm:mamba-130m-repro

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export MASTER_ADDR="localhost"
# export MASTER_PORT=54321

NAME="mamba-130m-repro"
IMAGE="mamba-ssm-fix.sqsh"
ROOT_DIR="/lustre/fsw/portfolios/adlr/users/bnorick"
CODE_DIR="${ROOT_DIR}/dev/experiments/mamba/code/megatron-lm"
IMAGES_DIR="${ROOT_DIR}/images"

RUN_DIR="${ROOT_DIR}/runs/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${RUN_DIR}/data-cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
IMAGE_PATH="${IMAGES_DIR}/${IMAGE}"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

DATA_PATH=/lustre/fsw/portfolios/llmservice/users/bnorick/data/datasets/pile/gpt-neox-20b

SEQ_LEN=2048
TRAIN_SAMPLES=146484375  # 300B tokens / 2048
LR_WARMUP_SAMPLES=3906252  # 0.026666679773866397 * 146484375
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES-LR_WARMUP_SAMPLES))

       # --train-iters 10000 \
       # --lr-decay-iters 320000 \
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

# single node test command
# run_cmd="torchrun --nproc_per_node 8 ${CODE_DIR}/pretrain_mamba.py ${options}"
run_cmd="python -u ${CODE_DIR}/pretrain_mamba.py ${options}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# echo srun -l --container-image "${IMAGE_PATH}" --container-mounts "/lustre:/lustre" --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" sh -c "${run_cmd}"
# exit
srun -l \
       --container-image "${IMAGE_PATH}" \
       --container-mounts "/lustre:/lustre" \
       --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
       sh -c "${run_cmd}"

set +x
