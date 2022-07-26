#!/bin/bash

NAME="gpt3-126m"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p ./logs

TENSORBOARD_DIR="./tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp-large/data/gpt3/gpt3_blend.sh

CHECKPOINT_DIR="./checkpoints/gpt3/${NAME}"

export LINEAR='{fi:{e:4,m:3,s:1,f:1,p:0},fw:{e:4,m:3,s:1,f:1,p:0},do:{e:4,m:3,s:1,f:1,p:0}}'

options=" \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --rampup-batch-size 32 32 1953125 \
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

#run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_gpt.py ${options}"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
	pretrain_gpt.py ${options}

#python ./pretrain_gpt.py ${options}

