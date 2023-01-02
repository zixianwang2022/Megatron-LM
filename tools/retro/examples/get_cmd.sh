#!/bin/bash

set -u

# DIR=$(dirname "$0")
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${DIR}/get_vars.sh

######## Data corpus. ########
# >>>
CORPUS="wiki"
RETRO_INDEX_STR="IVF262144_HNSW32,Flat"
RETRO_GPT_TRAIN_SAMPLES=2037248 LR_DECAY_SAMPLES=2 LR_WARMUP_SAMPLES=1
RETRO_EF_SEARCH=16 RETRO_NPROBE=4096
# +++
# CORPUS="corpus"
# RETRO_INDEX_STR="OPQ32_256,IVF4194304_HNSW32,PQ32"
# RETRO_GPT_TRAIN_SAMPLES=192000000 LR_DECAY_SAMPLES=166400000 LR_WARMUP_SAMPLES=162761
# RETRO_EF_SEARCH=32 RETRO_NPROBE=4096
# [x] ... RETRO_EF_SEARCH=256 RETRO_NPROBE=65536
# <<<

######## Repo. ########
REPO="retro"
# REPO="retro-batch"
# REPO="retro-embed"
# REPO="retro-wiki"
# REPO="retro-corpus"

######## Data blend. ########
. ${BLEND_SCRIPT_DIR}/gpt3_blend_${CORPUS}.sh
DATA_PATH=${DATA_BLEND}

######## Retro setup. ########
RETRO_WORKDIR=${RETRO_WORKDIRS}/${CORPUS}

# RETRO_TASKS="db-build"
# RETRO_TASKS="index-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
RETRO_TASKS="pretraining-query-nbrs"
# ... RETRO_TASKS="build" # ... the goal ...
# RETRO_TASKS="misc-index-update-training-block-size"
# RETRO_TASKS="misc-pretraining-test-retro-dataset"
# RETRO_TASKS="misc-pretraining-plot-acc"
# RETRO_TASKS="misc-pretraining-verify-nbrs"
# RETRO_TASKS="misc-index-remove-train-files,train"
# RETRO_TASKS="misc-index-remove-add-files,add"
# RETRO_TASKS="misc-index-verify"
# RETRO_TASKS="misc-index-verify-codes"
# RETRO_TASKS="misc-index-verify-nbrs"
# RETRO_TASKS="misc-index-time-hnsw"
# RETRO_TASKS="misc-index-time-query"
# RETRO_TASKS="misc-index-time-merge-partials"
# RETRO_TASKS="misc-db-nan-stats"
# RETRO_TASKS="misc-db-bert-nan-analysis"
# RETRO_TASKS="misc-db-print-embeddings"
# RETRO_TASKS="misc-db-print-neighbors"
# RETRO_TASKS="misc-index-megatron-huggingface-comparison-v0" # ?
# RETRO_TASKS="misc-index-megatron-huggingface-comparison-v1" # ?
# RETRO_TASKS="misc-index-megatron-huggingface-comparison-v2" # save to disk
# RETRO_TASKS="misc-index-megatron-huggingface-comparison-v3" # use merged valids
# RETRO_TASKS="misc-index-megatron-huggingface-comparison-v4" # use train embeds
# RETRO_TASKS="misc-index-megatron-huggingface-comparison-v5" # dist comparison
# RETRO_TASKS="misc-index-check-train-valid-split"
# RETRO_TASKS="misc-pretraining-compare-embeds"
# RETRO_TASKS="misc-pretraining-print-neighbors"
# RETRO_TASKS="misc-pretraining-compare-old-nbrs"

# RETRO_INDEX_TY=faiss-base
RETRO_INDEX_TY=faiss-par-add

# RETRO_PRECOMPUTE_BERT_LENGTHS
RETRO_GPT_SEQ_LENGTH=2048
RETRO_GPT_CHUNK_LENGTH=64
RETRO_GPT_MICRO_BATCH_SIZE=1 # *8
RETRO_GPT_GLOBAL_BATCH_SIZE=256
# RETRO_GPT_GLOBAL_BATCH_SIZE=512 # for debug.
RETRO_BERT_BATCH_SIZE=128 # optimal. [ mean seq length vs. batch size ]
RETRO_BERT_MAX_CHUNK_LENGTH=256
RETRO_NCHUNKS_SAMPLED=300000000
# RETRO_NCHUNKS_SAMPLED=3000000
RETRO_DOC_BLOCK_SIZE=100000
RETRO_BLOCK_SIZE=100000 # 10000, *100000, 1000000
# RETRO_INDEX_TRAIN_BLOCK_SIZE=3750000
RETRO_NNBRS_QUERY=2000
RETRO_NNBRS_TARGET=200
# RETRO_NNBRS_PRETRAINING=2

######## Megatron args. ########
SEED=1234 # default
DISTRIBUTED_TIMEOUT_MINUTES=600
# --no-load-rng \
MEGATRON_ARGS=" \
    --no-load-optim \
    --seed ${SEED} \
    --no-async-tensor-model-parallel-allreduce \
    --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES} \
    --tokenizer-type ${BERT_TOKENIZER} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size ${RETRO_GPT_MICRO_BATCH_SIZE} \
    --global-batch-size ${RETRO_GPT_GLOBAL_BATCH_SIZE} \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --train-samples ${RETRO_GPT_TRAIN_SAMPLES} \
    --load ${BERT_LOAD_PATH} \
    --data-path ${DATA_PATH} \
    --vocab-file ${BERT_VOCAB_FILE} \
    --data-impl mmap \
    --split 98,2,0 \
    --distributed-backend nccl \
    --lr 0.0001 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --eval-interval 2000 \
    --eval-iters 100 \
    --fp16 \
    --DDP-impl local \
    --dataloader-type cyclic \
    --no-data-sharding \
"

######## Retro args. ########
RETRO_ARGS=" \
    --output-bert-embeddings \
    \
    --retro-gpt-vocab-file ${GPT_VOCAB_FILE} \
    --retro-gpt-merge-file ${GPT_MERGE_FILE} \
    --retro-gpt-tokenizer-type GPT2BPETokenizer \
    --retro-gpt-seq-length ${RETRO_GPT_SEQ_LENGTH} \
    --retro-gpt-chunk-length ${RETRO_GPT_CHUNK_LENGTH} \
    --retro-bert-vocab-file ${BERT_VOCAB_FILE} \
    --retro-bert-tokenizer-type BertWordPieceLowerCase \
    --retro-bert-batch-size ${RETRO_BERT_BATCH_SIZE} \
    --retro-bert-max-chunk-length ${RETRO_BERT_MAX_CHUNK_LENGTH} \
    \
    --retro-tasks ${RETRO_TASKS} \
    --retro-index-ty ${RETRO_INDEX_TY} \
    --retro-index-str ${RETRO_INDEX_STR} \
    --retro-ef-search ${RETRO_EF_SEARCH} \
    --retro-nprobe ${RETRO_NPROBE} \
    \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-nchunks-sampled ${RETRO_NCHUNKS_SAMPLED} \
    --retro-doc-block-size ${RETRO_DOC_BLOCK_SIZE} \
    --retro-block-size ${RETRO_BLOCK_SIZE} \
    --retro-nnbrs-query ${RETRO_NNBRS_QUERY} \
    --retro-nnbrs-target ${RETRO_NNBRS_TARGET} \
    \
    --retro-return-doc-ids \
"

######## Command. ########
RETRO_PREPROCESS_CMD=" \
    ./tools/retro/main.py \
    ${MEGATRON_ARGS} \
    ${RETRO_ARGS} \
"

# eof.
