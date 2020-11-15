#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=<Specify path and file prefix>_text_sentence
CHECKPOINT_PATH=<Specify path>

python pretrain_t5.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --model-parallel-size 1 \
    --batch-size 4 \
    --seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --train-iters 2000000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --save ${CHECKPOINT_PATH} \
    --load ${CHECKPOINT_PATH} \
    --data-path ${DATA_PATH} \
    --vocab-file bert-vocab.txt \
    --vocab-extra-ids 100 \
    --data-impl mmap \
    --split 949,50,1 \
    --fp16 \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
 
