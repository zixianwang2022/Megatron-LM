#! /bin/bash

# Runs mamba model using the same hyper-params as the 345M transformer model

RANK=0
WORLD_SIZE=1


DATA_PATH=/preproc_data/Pile-CC_id_cleaned_shuf_text_document
ARTIFACTS_PATH=/artifacts
CHECKPOINT_PATH=/results/checkpoints

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR="localhost"
export MASTER_PORT=54321

python pretrain_mamba.py \
       --num-layers 48 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 4 \
       --seq-length 1024 \
       --max-position-embeddings 2048 \
       --train-iters 10000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-cache-path /cache \
       --vocab-file $ARTIFACTS_PATH/gpt2-vocab.json \
       --merge-file $ARTIFACTS_PATH/gpt2-merges.txt \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --bf16 \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_layer_spec \
       --normalization RMSNorm
