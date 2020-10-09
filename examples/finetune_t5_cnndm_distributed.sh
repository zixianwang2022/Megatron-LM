#!/bin/bash

#SBATCH -p luna -A adlr-nlp -t 4:00:00 --nodes=4 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=t5_main_220m_cnndm

BASE_DIR="/lustre/fsw/adlr-nlp/dsachan/"
DATA_DIR="${BASE_DIR}/data/cnndm"

TRAIN_DATA="${DATA_DIR}/train.source \
            ${DATA_DIR}/train.target"

VALID_DATA="${DATA_DIR}/val.source \
            ${DATA_DIR}/val.target"

TEST_DATA="${DATA_DIR}/test.source \
            ${DATA_DIR}/test.target"

PRETRAINED_CHECKPOINT="${BASE_DIR}/checkpoints/t5_main_full-wikipedia_base_mp1"
VOCAB_FILE="${BASE_DIR}/bert_vocab/bert-large-uncased-vocab.txt"
CHECKPOINT_PATH="${BASE_DIR}/checkpoints/t5_main_cnndm"

CONFIG_ARGS="--num-layers 12 \
             --hidden-size 768 \
             --num-attention-heads 12 \
             --kv-channels 64 \
             --ffn-hidden-size 3072 \
             --seq-length 512 \
             --decoder-seq-length 512 \
	           --vocab-extra-ids 100 \
             --max-position-embeddings 512 \
             --fp16 \
             --vocab-file $VOCAB_FILE \
             --model-parallel-size 1 \
             --num-workers 2 "

EXTRA_OPTIONS="--train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --test-data $TEST_DATA \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --save-interval 5000 \
               --save $CHECKPOINT_PATH \
               --log-interval 100 \
               --eval-interval 10000 \
               --eval-iters 10 \
               --weight-decay 1.0e-1"

OPTIONS=" \
       --distributed-backend nccl \
       --task CNNDM \
       --finetune \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 10 \
       --sample-rate 1.0 \
       --batch-size 4 \
       --eval-batch-size 12 \
       --beam-size 1 \
       --max-decode-len 512 \
       --lr 2e-5 \
       --warmup 0.0 \
       --no-load-optim \
       --no-load-rng \
       --DDP-impl local \
       --lr-decay-style linear "

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/tasks/main_t5.py ${OPTIONS} ${CONFIG_ARGS} ${EXTRA_OPTIONS}"

srun -l \
     --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/pytorch-faiss-gpu:20.07-py3-devel" \
     --container-mounts "/lustre/fsw/adlr-nlp:/lustre/fsw/adlr-nlp" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x
