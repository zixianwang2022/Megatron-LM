#!/bin/bash

NAME="bf16.gpt3.126m.linear.e4m3.f1.p0"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

TENSORBOARD_DIR="${DIR}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

DATA_DIR="/data/path/BookCorpus2_ftfy_cleaned_id_shuf_text_document"
CHECKPOINT_DIR="/home/ksivamani/fp8/megatron-lm/checkpoints/${NAME}"

cd megatron/fp/exmy
python setup.py build
cp build/lib*/*.so .
cd ../cudnn_funcs
python setup.py build
cp build/lib*/*.so .
cd ../histograms
python setup.py build
cp build/lib*/*.so .
cd ../../..

export LINEAR='{fi:{e:4,m:3,s:1,f:1,p:0,v:1},fw:{e:4,m:3,s:1,f:1,p:0,v:1},do:{e:4,m:3,s:1,f:1,p:0,v:1}}'

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
    --global-batch-size 16 \
    --train-iters 10 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_DIR} \
    --vocab-file /files/gpt2-vocab.json \
    --merge-file /files/gpt2-merges.txt \
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

#nsys profile --trace nvtx,cuda 
python -m torch.distributed.launch --use_env --nnodes=1 --nproc_per_node=8 ${DIR}/pretrain_gpt.py ${options}