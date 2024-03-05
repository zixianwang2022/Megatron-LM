#!/bin/bash

#SBATCH -p batch -A llmservice_nlp_fm -t 4:00:00 --nodes=8 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=llmservice_nlp_fm-yh:ct8x15b --array=1-30%1

export ADLR_SHARING=/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing

export OUTPUT=/lustre/fsw/coreai_dlalgo_llm/yihuih/moe

export SQSH=/lustre/fsw/portfolios/adlr/users/rprenger/sqsh

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b

NAME="ct15b"

DIR=/home/yihuih/llmservice/moe-mlm
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

INIT_CHECKPOINT_DIR="/lustre/fsw/coreai_dlalgo_llm/yihuih/checkpoints/15b/gpt3-15b-8t-tp8-pp8"

CHECKPOINT_DIR="${OUTPUT}/${NAME}"
RESET_STATE=""
if [[ ! -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    CHECKPOINT_DIR=$INIT_CHECKPOINT_DIR
    RESET_STATE="--reset-dataloader-state \
    --reset-lr-state \
    --override-opt_param-scheduler \
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
# . ${ADLR_SHARING}/nvllm-1.1t/data/tokens/multi-1.1t-gtc-blend-v0.1-localized.sh

. /lustre/fsw/coreai_dlalgo_llm/yihuih/nvllm-8t/8t.sh

options=" \
    --moe-grouped-gemm \
    --transformer-impl transformer_engine \
    --use-mcore-models \
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
    --pipeline-model-parallel-size 8 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --num-layers 32 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 1152 \
    --train-samples 195312500 \
    --lr-decay-samples 194921874 \
    --lr 4.5e-5 \
    --min-lr 4.5e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 200 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/share/llmservice_nlp_fm/adlr-nlp-sharing/nvllm-8t/utils/nemotron_2_256k.model \
    --data-path ${DATA_BLEND} \
    --data-cache-path ${DATA_CACHE} \
    --save-interval 10000 \
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

#  ([[ "\$SLURM_LOCALID" == "0" ]] && echo "installing" && pip install git+https://github.com/fanshiqing/grouped_gemm@main) ; ([[ "\$SLURM_LOCALID" != "0" ]] && echo "sleeping" && sleep 240) ;


run_cmd="
cd $DIR && python -u pretrain_gpt.py ${options}"

# srun --jobid=362390 -l --nodes=8 --ntasks-per-node=8     --container-image /lustre/fsw/coreai_dlalgo_llm/yihuih/images/24.01.sqsh      --container-mounts "/lustre:/lustre/,/home:/home"    bash -c "${run_cmd}"

srun -l \
     --container-image /lustre/fsw/coreai_dlalgo_llm/yihuih/images/24.01.sqsh \
     --container-mounts "/lustre:/lustre/,/home:/home" \
     --output=${LOG_DIR}/%x_%j_$DATETIME.log bash -c "${run_cmd}"

set +x