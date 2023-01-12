#!/bin/bash

set -u

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${DIR}/get_vars.sh

######## Data corpus. ########
CORPUS="wiki"
# CORPUS="corpus"

if [ "$CORPUS" = "wiki" ]; then
    RETRO_INDEX_STR="IVF262144_HNSW32,Flat"
    RETRO_GPT_TRAIN_SAMPLES=2037248
    LR_DECAY_SAMPLES=2
    LR_WARMUP_SAMPLES=1
    RETRO_GPT_EVAL_INTERVAL=2000
    RETRO_GPT_EVAL_ITERS=100
    RETRO_EF_SEARCH=16
    RETRO_NPROBE=4096
    BERT_EMBEDDER_TYPE=megatron
fi
if [ "$CORPUS" = "corpus" ]; then
    RETRO_INDEX_STR="OPQ32_256,IVF4194304_HNSW32,PQ32"
    RETRO_GPT_TRAIN_SAMPLES=192000000
    LR_DECAY_SAMPLES=166400000
    LR_WARMUP_SAMPLES=162761
    RETRO_GPT_EVAL_INTERVAL=2000
    RETRO_GPT_EVAL_ITERS=50
    RETRO_EF_SEARCH=32
    RETRO_NPROBE=4096
    # RETRO_EF_SEARCH=256
    # RETRO_NPROBE=65536
    BERT_EMBEDDER_TYPE=huggingface
fi

######## Repo. ########
REPO="retro"

######## Data blend. ########
. ${BLEND_SCRIPT_DIR}/gpt3_blend_${CORPUS}.sh
DATA_PATH=${DATA_BLEND}

######## Retro setup. ########
RETRO_WORKDIR=${RETRO_WORKDIRS}/${CORPUS}

# RETRO_TASKS="db-build"
# RETRO_TASKS="index-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
# RETRO_TASKS="pretraining-query-neighbors"

# (tasks below are less tested; for debugging)
# RETRO_TASKS="misc-index-verify-codes"
# RETRO_TASKS="misc-index-megatron-huggingface-comparison-full-db"
RETRO_TASKS="misc-index-megatron-huggingface-comparison-partial-db"
# RETRO_TASKS="misc-index-megatron-huggingface-comparison-neighbor-dists"
# RETRO_TASKS="misc-pretraining-verify-neighbors"
# RETRO_TASKS="misc-pretraining-print-neighbors"

# RETRO_INDEX_TY=faiss-base
RETRO_INDEX_TY=faiss-par-add

RETRO_GPT_SEQ_LENGTH=2048
RETRO_GPT_CHUNK_LENGTH=64
RETRO_GPT_MICRO_BATCH_SIZE=1 # *8
RETRO_GPT_GLOBAL_BATCH_SIZE=256
RETRO_BERT_BATCH_SIZE=128 # trade-off: mean seq length vs. batch size
RETRO_BERT_MAX_CHUNK_LENGTH=256
RETRO_NCHUNKS_SAMPLED=300000000
RETRO_DOC_BLOCK_SIZE=100000
RETRO_BLOCK_SIZE=100000
RETRO_NUM_NEIGHBORS_QUERY=2000
RETRO_NUM_NEIGHBORS_TARGET=200
# RETRO_NUM_NEIGHBORS_PRETRAINING=2

######## Megatron args. ########
SEED=1234 # default
DISTRIBUTED_TIMEOUT_MINUTES=600
# --no-load-rng \
# --no-initialization \
MEGATRON_ARGS=" \
    --no-load-optim \
    --exit-on-missing-checkpoint \
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
    --eval-interval ${RETRO_GPT_EVAL_INTERVAL} \
    --eval-iters ${RETRO_GPT_EVAL_ITERS} \
    --fp16 \
    --DDP-impl local \
    --dataloader-type cyclic \
    --no-data-sharding \
"

######## Retro args. ########
RETRO_ARGS=" \
    --bert-embedder-type ${BERT_EMBEDDER_TYPE} \
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
    --retro-num-neighbors-query ${RETRO_NUM_NEIGHBORS_QUERY} \
    --retro-num-neighbors-target ${RETRO_NUM_NEIGHBORS_TARGET} \
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
