#!/bin/bash


#SBATCH -p batch -A llmservice_nlp_fm -t 4:00:00 --nodes=8 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=llmservice_nlp_fm-yh:fixrouter_8x2b_upcycle_scattermoe_st_E16G16T16_clip2_scalerouter

export ADLR_SHARING=/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing

export OUTPUT=/home/yihuih/llmservice/moe

export SQSH=/lustre/fsw/portfolios/adlr/users/rprenger/sqsh

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b
export WANDB_CONSOLE="redirect"

NAME="fixrouter_8x2b_upcycle_scattermoe_st_E16G16T16_clip2_scalerouter"

DIR=/home/yihuih/llmservice/moe-mlm
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

INIT_CHECKPOINT_DIR="/home/yihuih/llmservice/fixrouter/2b_tp4_router001-te-scatter-E16G16T16"

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

DATA_CACHE="${OUTPUT}/data_cache"
mkdir -p ${DATA_CACHE}

# Get the data blend
. /home/yihuih/llmservice/data/1.1t.sh

options=" \
    --recompute-activations \
    --recompute-granularity full \
    --no-create-attention-mask-in-dataloader \
    --transformer-impl transformer_engine \
    --use-mcore-models \
    --use-distributed-optimizer \
    --num-experts 256 \
    --moe-router-topk 16 \
    --ffn-hidden-size 340 \
    --moe-router-type st \
    --moe-z-loss-coeff 1e-3 \
    --moe-aux-loss-coeff 1e-2 \
    --moe_log_load_balancing \
    --moe-scattermoe \
    --moe-scale-router 256.0 \
    --apply-layernorm-1p \
    --use-flash-attn \
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
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 512 \
    --train-samples 26855468 \
    --lr-decay-samples 25512695 \
    --lr-warmup-samples 25512 \
    --lr 2e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 1000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /home/yihuih/llmservice/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --data-cache-path ${DATA_CACHE} \
    --save-interval 500000 \
    --save ${OUTPUT}/${NAME} \
    --load ${CHECKPOINT_DIR} \
    --split 99,1,0 \
    --clip-grad 2.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
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

# 
# srun --jobid=469860 -N1 --tasks-per-node=8 --gpus-per-node=8 -l \
#      --container-image /home/yihuih/llmservice/images/24.01.sqsh \
#      --container-mounts "/lustre:/lustre/,/home:/home" \
#      bash -c "${run_cmd}"

# torchrun --nproc_per_node=8 pretrain_gpt.py ${options}

srun -l \
     --container-image /home/yihuih/llmservice/images/24.01.sqsh \
     --container-mounts "/lustre:/lustre/,/home:/home" \
     --no-container-mount-home bash -c "${run_cmd}"
set +x
