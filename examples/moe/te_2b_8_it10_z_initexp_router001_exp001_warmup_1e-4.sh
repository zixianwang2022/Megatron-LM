#!/bin/bash

##SBATCH -p batch_block1,batch_block3,batch_block4,adlr_services -A llmservice_nlp_fm -t 4:00:00 --nodes=2 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --gres=gpu:8 --dependency=singleton --job-name=llmservice_nlp_fm:te_2b_8_it10_z_initexp_router001_exp001_warmup_1e-4 --array=1-30%1

#SBATCH -p batch -A llmservice_nlp_fm -t 4:00:00 --nodes=16 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=llmservice_nlp_fm-yh:te_2b_8_it10_z_initexp_router001_exp001_warmup_1e-4 --array=1-20%1

export ADLR_SHARING=/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing

export OUTPUT=/home/yihuih/llmservice/moe

export SQSH=/lustre/fsw/portfolios/adlr/users/rprenger/sqsh

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b

NAME="te_2b_8_it10_z_initexp_router001_exp001_warmup_1e-4"

DIR=/home/yihuih/llmservice/moe-mlm
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

INIT_CHECKPOINT_DIR="/home/yihuih/llmservice/moe-init/gpt3-8x2b_TP8_router001_exp001"

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
    --transformer-impl transformer_engine \
    --use-mcore-models \
    --use-distributed-optimizer \
    --num-experts 8 \
    --moe-z-loss-coeff 1e-3 \
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
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --train-samples 26855468 \
    --lr-decay-samples 25512695 \
    --lr-warmup-samples 25512 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 500 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /home/yihuih/llmservice/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
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

srun -l \
     --container-image /home/yihuih/llmservice/images/24.01.sqsh \
     --container-mounts "/lustre:/lustre/,/home:/home" \
     --output=${LOG_DIR}/%x_%j_$DATETIME.log bash -c "${run_cmd}"
set +x
