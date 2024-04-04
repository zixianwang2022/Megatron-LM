#!/bin/bash

#SBATCH -p batch -A llmservice_nlp_fm -t 4:00:00 --nodes=16 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=8t-8x15b_E64G8T16 --array=1-30%1

export ADLR_SHARING=/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing

export OUTPUT=/lustre/fsw/coreai_dlalgo_llm/yihuih/moe

export SQSH=/lustre/fsw/portfolios/adlr/users/rprenger/sqsh

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b

NAME="8t-8x15b_E64G8T16"

DIR=/home/yihuih/llmservice/moe-mlm
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`


CHECKPOINT_DIR="${OUTPUT}/${NAME}"
RESET_STATE=""

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
    --num-experts 64 \
    --moe-router-topk 8 \
    --moe-group-size 8 \
    --ffn-hidden-size 3072 \
    --moe-grouped-gemm \
    --moe-router-type grouped \
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
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 8 \
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
    --lr-warmup-samples 391680 \
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

