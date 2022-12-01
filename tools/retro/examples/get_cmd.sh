#!/bin/bash

set -u

# echo "SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
# NPROCS=$SLURM_TASKS_PER_NODE
# >>>
NPROCS=1
# NPROCS=2
# NPROCS=4
# NPROCS=8
# NPROCS=16
# NPROCS=128
# >>>

PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-retro-process
# CORPUS="play"
CORPUS="wiki"
# CORPUS="corpus"

# Data blend.
# . /gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/gpt3_blend.sh
. /gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/misc/gpt3_blend_${CORPUS}.sh
DATA_PATH=${DATA_BLEND}

GPT_VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json
GPT_MERGE_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt
GPT_TOKENIZER_TYPE=GPT2BPETokenizer

BERT_LOAD_PATH=/home/universal-lm-data-netapp/chkpts/bert/345M_no_rng
BERT_VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt
BERT_TOKENIZER_TYPE=BertWordPieceLowerCase

# >>>>>>>>>>>>>>>>>>>>>>>
RETRO_WORKDIR=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/${CORPUS}

# RETRO_TASKS="db-build"
# RETRO_TASKS="db-preprocess"
# [x] ... RETRO_TASKS="db-embed"
# RETRO_TASKS="index-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
# RETRO_TASKS="index-remove-train-files,train"
# RETRO_TASKS="index-remove-add-files,add"
# RETRO_TASKS="index-verify"
# RETRO_TASKS="index-verify-codes"
# RETRO_TASKS="index-verify-nbrs"
# RETRO_TASKS="pretraining-query-nbrs"
# ... RETRO_TASKS="build" # ... the goal ...
# RETRO_TASKS="misc-pretraining-test-retro-dataset"
# RETRO_TASKS="misc-pretraining-plot-acc"
# RETRO_TASKS="misc-pretraining-verify-nbrs"
# RETRO_TASKS="misc-index-time-hnsw"
# RETRO_TASKS="misc-index-time-query"
# RETRO_TASKS="misc-index-time-merge-partials"
# RETRO_TASKS="misc-db-nan-stats"
# RETRO_TASKS="misc-db-bert-nan-analysis"
# RETRO_TASKS="misc-index-megatron-huggingface-comparison"
# RETRO_TASKS="misc-index-check-train-valid-split"
RETRO_TASKS="misc-pretraining-print-neighbors"
# RETRO_TASKS="misc-pretraining-compare-old-nbrs"

# RETRO_INDEX_TY=faiss-base
RETRO_INDEX_TY=faiss-par-add
# RETRO_INDEX_TY=faiss-decomp

# >>>
# RETRO_NCLUSTERS=4194304
# # RETRO_NCLUSTERS=32768 # for 169320 training samples
# RETRO_HNSW_M=32
# RETRO_PQ_M=32
# RETRO_IVF_DIM=256
# +++
RETRO_INDEX_STR="IVF262144_HNSW32,Flat"
RETRO_GPT_TRAIN_SAMPLES=2037248 LR_DECAY_SAMPLES=1 LR_WARMUP_SAMPLES=1
# +++
# RETRO_INDEX_STR="OPQ32_256,IVF4194304_HNSW32,PQ32"
# TRAIN_SAMPLES=192000000 LR_DECAY_SAMPLES=166400000 LR_WARMUP_SAMPLES=162761
# <<<

RETRO_EF_SEARCH=32 # 256
RETRO_NPROBE=4096 # 65536

# RETRO_PRECOMPUTE_BERT_LENGTHS
RETRO_GPT_SEQ_LENGTH=2048
RETRO_GPT_CHUNK_LENGTH=64
RETRO_GPT_MICRO_BATCH_SIZE=8
RETRO_GPT_GLOBAL_BATCH_SIZE=256
RETRO_BERT_BATCH_SIZE=128 # optimal. [ mean seq length vs. batch size ]
RETRO_BERT_MAX_CHUNK_LENGTH=256
RETRO_NCHUNKS_SAMPLED=300000000
# RETRO_NCHUNKS_SAMPLED=3000000
RETRO_DOC_BLOCK_SIZE=100000
RETRO_BLOCK_SIZE=100000 # 10000, *100000, 1000000
RETRO_NNBRS_QUERY=2000
RETRO_NNBRS_TARGET=200
RETRO_NNBRS_PRETRAINING=2

SEED=1234 # default
# SEED=1001
DISTRIBUTED_TIMEOUT_MINUTES=600 # 180
# MICRO_BATCH_SIZE=128 # optimal. [ mean seq length vs. batch size ]

#     --log-interval 100 \
#     --save-interval 10000 \
#     --cyclic-train-iters 750000 \ # ... retro pretraining only
MEGATRON_ARGS=" \
    --seed ${SEED} \
    --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES} \
    --tokenizer-type ${BERT_TOKENIZER_TYPE} \
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

# --retro-precompute-bert-lengths \
# --retro-embedder ${RETRO_EMBEDDER} \
# --retro-dump-huggingface-embeddings \
# --retro-nclusters ${RETRO_NCLUSTERS} \
# --retro-ivf-dim ${RETRO_IVF_DIM} \
# --retro-hnsw-m ${RETRO_HNSW_M} \
# --retro-pq-m ${RETRO_PQ_M} \
RETRO_ARGS=" \
    --output-bert-embeddings \

    --retro-gpt-vocab-file ${GPT_VOCAB_FILE} \
    --retro-gpt-merge-file ${GPT_MERGE_FILE} \
    --retro-gpt-tokenizer-type ${GPT_TOKENIZER_TYPE} \
    --retro-gpt-seq-length ${RETRO_GPT_SEQ_LENGTH} \
    --retro-gpt-chunk-length ${RETRO_GPT_CHUNK_LENGTH} \
    --retro-bert-vocab-file ${BERT_VOCAB_FILE} \
    --retro-bert-tokenizer-type ${BERT_TOKENIZER_TYPE} \
    --retro-bert-batch-size ${RETRO_BERT_BATCH_SIZE} \
    --retro-bert-max-chunk-length ${RETRO_BERT_MAX_CHUNK_LENGTH} \

    --retro-tasks ${RETRO_TASKS} \
    --retro-index-ty ${RETRO_INDEX_TY} \
    --retro-index-str ${RETRO_INDEX_STR} \
    --retro-ef-search ${RETRO_EF_SEARCH} \
    --retro-nprobe ${RETRO_NPROBE} \

    --retro-workdir ${RETRO_WORKDIR} \
    --retro-nchunks-sampled ${RETRO_NCHUNKS_SAMPLED} \
    --retro-doc-block-size ${RETRO_DOC_BLOCK_SIZE} \
    --retro-block-size ${RETRO_BLOCK_SIZE} \
    --retro-nnbrs-query ${RETRO_NNBRS_QUERY} \
    --retro-nnbrs-target ${RETRO_NNBRS_TARGET} \
    --retro-nnbrs-pretraining ${RETRO_NNBRS_PRETRAINING} \

    --retro-return-doc-ids \
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
