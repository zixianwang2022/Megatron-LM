#!/bin/bash

##SBATCH -p interactive,luna -A adlr-nlp -t 1:00:00 --time-min=1:00:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=megatron_gpt2_345m_mp2_baseline_oct24_lambada_nov12
DIR=`pwd`
OUTPUT_DIR=./eval #/lustre/fsw/adlr-nlp/jcasper/output/debug/
NAME=eval_gpt2
LOGDIR=${OUTPUT_DIR}/logs/${NAME}
CKPTDIR=${OUTPUT_DIR}/checkpoints/${NAME}
TBDIR=${OUTPUT_DIR}/runs/${NAME}

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

SOURCE_DIR=${DIR}
SCRIPTS=${DIR}

# LAMBADA Zero-Shot Task
TASK="LAMBADA"
VALID_DATA=/lustre/fsw/adlr/adlr-nlp/mpatwary/data/zeroshot_eval/lambada_test.jsonl

# WikiText-103 Zero-Shot Task
#TASK="WIKITEXT103"
#VALID_DATA=/lustre/fsw/adlr/adlr-nlp/mpatwary/data/zeroshot_eval/wikitext-103/wiki.test.tokens

VOCAB_FILE=/lustre/fsw/adlr/adlr-nlp-large/data/bpe/gpt2-vocab.json
MERGE_FILE=/lustre/fsw/adlr/adlr-nlp-large/data/bpe/gpt2-merges.txt

# BF16 Checkpoint
#CHECKPOINT=/lustre/fsw/gpu-comparch/dstosic/fp8/adlr-nlp-large/checkpoints/bf16.gpt3.126m/

# Fp8 Checkpoint
CHECKPOINT=/lustre/fsw/gpu-comparch/dstosic/fp8/adlr-nlp-large/checkpoints/floatscale.bf16.gpt3.126m.linear.e4m3.f128.p0/

# Fp8 Meta Data
export LINEAR='{fi:{e:4,m:3,s:1,f:0},fw:{e:4,m:3,s:1,f:0},do:{e:4,m:3,s:1,f:0}}'

# FP32 Meta Data
#export LINEAR='{fi:{e:8,m:23,s:0,f:0},fw:{e:8,m:23,s:0,f:0},do:{e:8,m:23,s:0,f:0}}'

ARGS=" \
               --task $TASK \
               --valid-data $VALID_DATA \
               --tokenizer-type GPT2BPETokenizer \
               --vocab-file $VOCAB_FILE \
               --strict-lambada \
               --merge-file $MERGE_FILE \
               --load $CHECKPOINT \
               --tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 1 \
               --num-layers 12 \
               --hidden-size 768 \
               --num-attention-heads 12 \
               --micro-batch-size 4 \
               --checkpoint-activations \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --log-interval 10 \
               --bf16 \
               --no-load-optim \
               --no-load-rng"


${SCRIPTS}/bind.sh --cpu=${SCRIPTS}/dgxa100_ccx.sh --mem=${SCRIPTS}/dgxa100_ccx.sh \
          python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
          ${SOURCE_DIR}/tasks/main.py \
          $ARGS
set +x

