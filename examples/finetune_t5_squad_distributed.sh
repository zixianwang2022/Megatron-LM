#! /bin/bash

# This scripts finetune and evaluate t5 models on SQuAD 1.1 task

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="data/SQuAD/train-v1.1.json"
VALID_DATA="data/SQuAD/dev-v1.1.json"

PRETRAINED_CHECKPOINT="checkpoints/t5_base"
VOCAB_FILE="bert-large-uncased-vocab.txt"
CHECKPOINT_PATH="checkpoints/t5_base_squad"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main_t5.py \
            --task SQUAD \
            --seed 4567 \
            --train-data $TRAIN_DATA \
            --valid-data $VALID_DATA \
            --tokenizer-type BertWordPieceLowerCase \
            --vocab-file $VOCAB_FILE \
            --epochs 3 \
            --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
            --model-parallel-size 1 \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --kv-channels 64 \
            --ffn-hidden-size 3072 \
            --batch-size 4 \
            --eval-batch-size 8 \
            --beam-size 1 \
            --lr 2.0e-5 \
            --warmup 0.00 \
            --max-decode-len 512 \ 
            --seq-length 512 \
            --decoder-seq-length 128 \
            --vocab-extra-ids 100 \
            --max-position-embeddings 512 \
            --fp16 \
            --save-interval 5000 \
            --log-interval 100 \
            --eval-interval 10000 \
            --eval-iters 10 \
            --weight-decay 1.0e-2 \
            --finetune \
            --no-load-optim \
            --no-load-rng"
 

