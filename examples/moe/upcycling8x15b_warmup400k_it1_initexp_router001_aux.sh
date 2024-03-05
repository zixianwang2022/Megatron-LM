#!/bin/bash

#SBATCH -p batch_block1,batch_block3,batch_block4 -A llmservice_nlp_fm -t 4:00:00 --nodes=32 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=llmservice_nlp_fm-yh:upcycling8x15b_warmup400k_it1_initexp_router001_aux --array=1-10%1

# adlr_services: QoS=5_nodes_max
# ,interactive,batch_singlenode
# batch,backfill,hp

export OUTPUT=/home/yihuih/llmservice/moe

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b

NAME="upcycling8x15b_warmup400k_it1_initexp_router001_aux"

DIR=/home/yihuih/llmservice/moe-mlm
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

INIT_CHECKPOINT_DIR="/home/yihuih/llmservice/moe-init/gpt3-8x15b-8t-tp8-pp8_init001"

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


. /home/yihuih/llmservice/data/8t.sh

options=" \
    --num-experts 8 \
    --moe-z-loss-coeff 1e-3 \
    --moe-aux-loss-coeff 1e-2 \
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
    --exit-signal-handler \
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
    --train-samples 19531250 \
    --lr-decay-samples 19492187 \
    --lr-warmup-samples 390625 \
    --lr 4.5e-4 \
    --min-lr 4.5e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 200 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /home/yihuih/llmservice/data/nemotron_2_256k.model \
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

# srun --jobid=369250 -l --nodes=8 --ntasks-per-node=8     --container-image /home/yihuih/llmservice/images/24.01.sqsh      --container-mounts "/lustre:/lustre/,/home:/home"    bash -c "${run_cmd}"

srun -l \
     --container-image /home/yihuih/llmservice/images/24.01.sqsh \
     --container-mounts "/lustre:/lustre/,/home:/home" \
     --output=${LOG_DIR}/%x_%j_$DATETIME.log bash -c "${run_cmd}"

set +x
