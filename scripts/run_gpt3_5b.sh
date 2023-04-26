#!/bin/bash

#SBATCH -p luna -A gpu-comparch -t 4:00:00 --nodes=20 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=gpu-comparch-psx-5b:recipe.bf16.gpt3.5b.linear.e4m3.f32.p0.history

NAME="recipe.bf16.gpt3.5b.linear.hybrid.f32.p0.history"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

TENSORBOARD_DIR="${DIR}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp-large/data/gpt3/gpt3_blend.sh

CHECKPOINT_DIR="/lustre/fsw/gpu-comparch/dstosic/fp8/adlr-nlp-large/checkpoints/${NAME}"

export LINEAR='{fi:{e:4,m:3,s:1,f:32,p:0},fw:{e:4,m:3,s:1,f:32,p:0},do:{e:4,m:3,s:1,f:32,p:0}}'
export RECIPE='history'

BPE_DIR="/lustre/fsw/adlr/adlr-nlp-large/data/bpe"

options=" \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 1280 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 244141 \
    --lr 1.2e-4 \
    --min-lr 1.2e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --vocab-file ${BPE_DIR}/gpt2-vocab.json \
    --merge-file ${BPE_DIR}/gpt2-merges.txt \
    --save-interval 1000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 9999,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --no-async-tensor-model-parallel-allreduce"
#   --rampup-batch-size 32 32 2929688 \

run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_gpt.py ${options}"

srun -l \
     --container-image "/lustre/fsw/adlr/adlr-nlp/images/pytorch+bf16_nccl_fusion.sqsh" \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr,/lustre/fsw/gpu-comparch:/lustre/fsw/gpu-comparch" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x
