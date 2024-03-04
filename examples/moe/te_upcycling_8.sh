#!/bin/bash

##SBATCH -p batch_block1,batch_block2,interactive -A llmservice_nlp_fm -t 0:30:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --gres=gpu:8 --dependency=singleton --job-name=llmservice_nlp_fm:843m_continue_1e5_experts_4

#SBATCH -p batch_block1,batch_block2 -A llmservice_nlp_fm -t 4:00:00 --nodes=8 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --gres=gpu:8 --dependency=singleton --job-name=llmservice_nlp_fm:te_843m_continue_1e5_experts_8 --array=1-10%1


export ADLR_SHARING=/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing

export OUTPUT=/lustre/fsw/portfolios/llmservice/users/yihuih/moe

export SQSH=/lustre/fsw/portfolios/adlr/users/rprenger/sqsh

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b

NAME="te_843m_continue_1e5_experts_8"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

INIT_CHECKPOINT_DIR="/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/checkpoints/mcore-version/843m"

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
. ${ADLR_SHARING}/nvllm-1.1t/data/tokens/multi-1.1t-gtc-blend-v0.1-localized.sh

options=" \
    --transformer-impl transformer_engine \
    --use-mcore-models \
    --moe-grouped-gemm \
    --num-experts 8 \
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
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 128 \
    --train-samples 600000000 \
    --lr 1e-5 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model $ADLR_SHARING/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --data-cache-path ${DATA_CACHE} \
    --save-interval 20000 \
    --save ${OUTPUT}/${NAME} \
    --load ${CHECKPOINT_DIR} \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --wandb-project upcycling \
    --wandb-exp-name $NAME $RESET_STATE
"

# ([[ "\$SLURM_LOCALID" == "0" ]] && echo "installing" && pip install wandb) ; ([[ "\$SLURM_LOCALID" != "0" ]] && echo "sleeping" && sleep 30) ;
run_cmd="
cd $DIR && python -u pretrain_gpt.py ${options}"
#  --jobid=451511 -N1 --gpus-per-node=8
srun -l \
     --container-image /lustre/fsw/portfolios/llmservice/users/yihuih/images/24.01.sqsh \
     --container-mounts "/lustre:/lustre/,/home:/home" \
     --output=${LOG_DIR}/%x_%j_$DATETIME.log bash -c "${run_cmd}"
set +x
