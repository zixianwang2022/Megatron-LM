#!/bin/bash

set -u

# echo "SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
# NPROCS=$SLURM_TASKS_PER_NODE
# >>>
# NPROCS=1
# NPROCS=2
# NPROCS=4
# NPROCS=8
NPROCS=16
# NPROCS=128
# >>>

PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-retro-preprocess

# Data blend.
# . /gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/gpt3_blend.sh
. /gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/preprocess/gpt3_blend.sh
DATA_PATH=${DATA_BLEND}

GPT_VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json
GPT_MERGE_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt
GPT_TOKENIZER_TYPE=GPT2BPETokenizer

BERT_LOAD_PATH=/home/universal-lm-data-netapp/chkpts/bert/345M_no_rng
BERT_VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt
BERT_TOKENIZER_TYPE=BertWordPieceLowerCase

# >>>>>>>>>>>>>>>>>>>>>>>
RETRO_WORKDIR=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/1

# RETRO_PROFILE_STAGE_STOP="preprocess"
# RETRO_PROFILE_STAGE_STOP="cluster"

# RETRO_TASKS="db-build"
RETRO_TASKS="db-preprocess"
# RETRO_TASKS="db-embed"
# RETRO_TASKS="index-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
# RETRO_TASKS="index-remove-train-files,train"
# RETRO_TASKS="index-remove-add-files,add"
# RETRO_TASKS="index-verify"
# RETRO_TASKS="index-verify-codes"
# RETRO_TASKS="index-verify-nbrs"
# RETRO_TASKS="pretraining-build-nbrs"
# RETRO_TASKS="pretraining-embed-chunks"
# RETRO_TASKS="pretraining-query-nbrs"
# RETRO_TASKS="pretraining-test-retro-dataset"
# RETRO_TASKS="pretraining-plot-acc"
# RETRO_TASKS="pretraining-verify-nbrs"
# RETRO_TASKS="misc-time-hnsw"
# RETRO_TASKS="misc-time-query"
# RETRO_TASKS="misc-time-merge-partials"
# RETRO_TASKS="misc-copy-corpus-dirty"
# RETRO_TASKS="misc-nan-stats"
# RETRO_TASKS="misc-bert-nan-analysis"
# RETRO_TASKS="build" # ... the goal.

RETRO_INDEX_TY=faiss-base
# RETRO_INDEX_TY=faiss-par-add
# RETRO_INDEX_TY=faiss-decomp

# RETRO_NCLUSTERS=4194304
RETRO_NCLUSTERS=32768 # for 169320 training samples
RETRO_HNSW_M=32
RETRO_PQ_M=32
RETRO_IVF_DIM=256

RETRO_EF_SEARCH=256
RETRO_NPROBE=65536

# RETRO_PRECOMPUTE_BERT_LENGTHS
RETRO_GPT_SEQ_LENGTH=2048
RETRO_GPT_CHUNK_LENGTH=64
RETRO_BERT_MAX_CHUNK_LENGTH=256
# RETRO_NCHUNKS_SAMPLED=300000000
RETRO_NCHUNKS_SAMPLED=3000000
RETRO_BLOCK_SIZE=100000 # 10000, *100000, 1000000
RETRO_NNBRS_QUERY=2000
RETRO_NNBRS_TARGET=200
RETRO_NNBRS_PRETRAINING=2

SEED=1001
DISTRIBUTED_TIMEOUT_MINUTES=600 # 180
# MICRO_BATCH_SIZE=1024 # oom
# MICRO_BATCH_SIZE=512
# MICRO_BATCH_SIZE=256
MICRO_BATCH_SIZE=128 # optimal. [ mean seq length vs. batch size ]
# MICRO_BATCH_SIZE=64
# MICRO_BATCH_SIZE=32
# MICRO_BATCH_SIZE=16 # good
# MICRO_BATCH_SIZE=8
# MICRO_BATCH_SIZE=4

MEGATRON_ARGS=" \
    --seed ${SEED} \
    --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES} \
    --tokenizer-type ${BERT_TOKENIZER_TYPE} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --train-samples 192000000 \
    --load ${BERT_LOAD_PATH} \
    --data-path ${DATA_PATH} \
    --vocab-file ${BERT_VOCAB_FILE} \
    --data-impl mmap \
    --split 98,2,0 \
    --distributed-backend nccl \
    --lr 0.0001 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --fp16 \
"

# --retro-precompute-bert-lengths \
RETRO_ARGS=" \
    --retro-gpt-vocab-file ${GPT_VOCAB_FILE} \
    --retro-gpt-merge-file ${GPT_MERGE_FILE} \
    --retro-gpt-tokenizer-type ${GPT_TOKENIZER_TYPE} \
    --retro-gpt-seq-length ${RETRO_GPT_SEQ_LENGTH} \
    --retro-gpt-chunk-length ${RETRO_GPT_CHUNK_LENGTH} \
    --retro-bert-vocab-file ${BERT_VOCAB_FILE} \
    --retro-bert-tokenizer-type ${BERT_TOKENIZER_TYPE} \
    --retro-bert-max-chunk-length ${RETRO_BERT_MAX_CHUNK_LENGTH} \

    --retro-tasks ${RETRO_TASKS} \
    --retro-index-ty ${RETRO_INDEX_TY} \
    --retro-nclusters ${RETRO_NCLUSTERS} \
    --retro-ivf-dim ${RETRO_IVF_DIM} \
    --retro-hnsw-m ${RETRO_HNSW_M} \
    --retro-pq-m ${RETRO_PQ_M} \
    --retro-ef-search ${RETRO_EF_SEARCH} \
    --retro-nprobe ${RETRO_NPROBE} \

    --retro-workdir ${RETRO_WORKDIR} \
    --retro-nchunks-sampled ${RETRO_NCHUNKS_SAMPLED} \
    --retro-block-size ${RETRO_BLOCK_SIZE} \
    --retro-nnbrs-query ${RETRO_NNBRS_QUERY} \
    --retro-nnbrs-target ${RETRO_NNBRS_TARGET} \
    --retro-nnbrs-pretraining ${RETRO_NNBRS_PRETRAINING} \
"

RETRO_PREPROCESS_CMD=" \
    python -m torch.distributed.launch \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ./tools/retro/main.py \
    ${MEGATRON_ARGS} \
    ${RETRO_ARGS} \
"

# eof
