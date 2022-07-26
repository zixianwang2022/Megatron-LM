#!/bin/bash

#SBATCH -p luna -A gpu-comparch -t 4:00:00 --nodes=4 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=gpu-comparch-psx:bf16.gpt3.126m

NAME="bf16.gpt3.126m"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

TENSORBOARD_DIR="${DIR}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp-large/data/gpt3/gpt3_blend.sh

CHECKPOINT_DIR="/lustre/fsw/gpu-comparch/dstosic/fp8/adlr-nlp-large/checkpoints/${NAME}"

export LINEAR='{fi:{e:8,m:23,s:0,f:0,p:0},fw:{e:8,m:23,s:0,f:0,p:0},do:{e:8,m:23,s:0,f:0,p:0}}'

options=" \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --vocab-file /lustre/fsw/adlr/adlr-nlp-large/data/bpe/gpt2-vocab.json \
    --merge-file /lustre/fsw/adlr/adlr-nlp-large/data/bpe/gpt2-merges.txt \
    --save-interval 10000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.023 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR} "
    #--rampup-batch-size 32 32 1953125 \

run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_gpt.py ${options}"

srun -l \
     --container-image "/lustre/fsw/adlr/adlr-nlp/images/pytorch+bf16_nccl_fusion.sqsh" \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr,/lustre/fsw/gpu-comparch:/lustre/fsw/gpu-comparch" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x
