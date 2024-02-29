#!/bin/bash

#SBATCH -p batch_block1,batch_block2 -A llmservice_nlp_fm -t 0:30:00 --nodes=8 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --gres=gpu:8 --dependency=singleton --job-name=llmservice_nlp_fm:upcycling8x15b

# --array=1-10%1

export ADLR_SHARING=/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing

export OUTPUT=/lustre/fsw/portfolios/llmservice/users/yihuih/moe

export SQSH=/lustre/fsw/portfolios/adlr/users/rprenger/sqsh

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b

NAME="upcycling8x15b"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

# INIT_CHECKPOINT_DIR="/lustre/fs3/portfolios/adlr/users/rprenger/moe/843m_converted_8_experts"

# CHECKPOINT_DIR="${OUTPUT}/${NAME}"
# RESET_STATE=""
# if [[ ! -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
#     CHECKPOINT_DIR=$INIT_CHECKPOINT_DIR
#     RESET_STATE="--reset-dataloader-state \
#     --override-opt_param-scheduler \
#     --reset-lr-state"
# fi

    # --load ${CHECKPOINT_DIR} \
    # --data-path ${DATA_BLEND} \
    # --data-cache-path ${DATA_CACHE} \

LOG_DIR="${OUTPUT}/${NAME}/logs"
mkdir -p ${LOG_DIR}
TENSORBOARD_DIR="${OUTPUT}/${NAME}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}

DATA_CACHE="${OUTPUT}/data_cache"
mkdir -p ${DATA_CACHE}

# Get the data blend
. ${ADLR_SHARING}/nvllm-1.1t/data/tokens/multi-1.1t-gtc-blend-v0.1-localized.sh

options=" \
    --num-experts 8 \
    --use-mcore-models \
    --no-load-optim \
    --num-experts 8 \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --squared-relu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 8 \
    --expert-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --train-samples 600000000 \
    --lr 4.5e-4 \
    --min-lr 4.5e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model ${ADLR_SHARING}/nvllm-8t/data/tokens-shuffle/utils/nemotron_2_256k.model \
    --mock-data \
    --save-interval 20000 \
    --save ${OUTPUT}/${NAME} \
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

# ([[ "\$SLURM_LOCALID" == "0" ]] && echo "installing" && pip install wandb) ; ([[ "\$SLURM_LOCALID" != "0" ]] && echo "sleeping" && sleep 30) ;

run_cmd="cd $DIR && python -u pretrain_gpt.py ${options}"

srun -l \
     --container-image "/lustre/fsw/portfolios/llmservice/users/yihuih/images/nemo:24.01.framework.sqsh" \
     --container-mounts "/lustre:/lustre/,/home:/home" \
     --output=${LOG_DIR}/%x_%j_$DATETIME.log bash -c "${run_cmd}"
set +x