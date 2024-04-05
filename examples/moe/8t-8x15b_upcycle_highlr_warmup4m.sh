#!/bin/bash

#SBATCH -p batch -A llmservice_nlp_fm -t 4:00:00 --nodes=16 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=8t-8x15b_upcycle_highlr_warmup4m --array=1-30%1

export ADLR_SHARING=/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing

export OUTPUT=/lustre/fsw/coreai_dlalgo_llm/yihuih/moe

export SQSH=/lustre/fsw/portfolios/adlr/users/rprenger/sqsh

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b

NAME="8t-8x15b_upcycle_highlr_warmup4m"

DIR=/home/yihuih/llmservice/moe-mlm
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

INIT_CHECKPOINT_DIR="/home/yihuih/llmservice/moe-init/15b/gpt3-15b-8t-tp4-pp4_router001-te-gg-st4xw2"

CHECKPOINT_DIR="${OUTPUT}/${NAME}"
RESET_STATE=""
if [[ ! -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    CHECKPOINT_DIR=$INIT_CHECKPOINT_DIR
    RESET_STATE="--reset-dataloader-state \
    --override-opt_param-scheduler \
    --reset-lr-state \
    --no-load-rng \
    --no-load-optim
"
fi

LOG_DIR="${OUTPUT}/${NAME}/logs"
mkdir -p ${LOG_DIR}
TENSORBOARD_DIR="${OUTPUT}/${NAME}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}

DATA_CACHE="${OUTPUT}/data_cache-8t"
mkdir -p ${DATA_CACHE}

# Get the data blend
# . /home/yihuih/llmservice/data/8t.sh

. /lustre/fsw/coreai_dlalgo_llm/yihuih/nvllm-8t/8t.sh

options=" \
    --global-batch-size 2304 \
    --transformer-impl transformer_engine \
    --use-mcore-models \
    --moe-grouped-gemm \
    --num-experts 8 \
    --moe-router-type st \
    --moe-z-loss-coeff 1e-3 \
    --moe-aux-loss-coeff 1e-2 \
    --moe_log_load_balancing \
    --use-distributed-optimizer \
    --apply-layernorm-1p \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --squared-relu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 4 \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --train-samples 585937500 \
    --lr-decay-samples 584765624 \
    --lr-warmup-samples 3916800 \
    --lr-warmup-init 4.5e-5 \
    --lr 4.5e-4 \
    --min-lr 4.5e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 1000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/share/llmservice_nlp_fm/adlr-nlp-sharing/nvllm-8t/utils/nemotron_2_256k.model \
    --data-path ${DATA_BLEND} \
    --data-cache-path ${DATA_CACHE} \
    --save-interval 500000 \
    --save ${OUTPUT}/${NAME} \
    --load ${CHECKPOINT_DIR} \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.0134 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --bf16 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --wandb-project upcycling \
    --wandb-exp-name $NAME $RESET_STATE
"

run_cmd="
cd $DIR && python -u pretrain_gpt.py ${options}"



srun -l \
     --container-image /home/yihuih/llmservice/images/24.01.sqsh \
     --container-mounts "/lustre:/lustre/,/home:/home" \
      bash -c "${run_cmd}"

set +x

