#!/bin/bash

REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-retro-dedupe-sqlite"
RETRO_WORKDIR="/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/wiki-tiny"

# RETRO_TASKS="db-build"
RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
# RETRO_TASKS="pretraining-query-neighbors"

# RETRO_INDEX_STR="IVF262144_HNSW32,Flat"
RETRO_INDEX_STR="IVF65536_HNSW8,Flat"

# RETRO_INDEX_TRAIN_SAMPLES=10000000
RETRO_INDEX_TRAIN_SAMPLES=1000000
RETRO_INDEX_TRAIN_LOAD_FRACTION=1.0
RETRO_INDEX_ADD_LOAD_FRACTION=1.0

DATA_BLEND="1.000 /lustre/fs1/portfolios/adlr/users/lmcafee/corpus-530b/wiki-tiny/wiki-200k_text_document"

ARGS=" \
    --distributed-timeout-minutes 600 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --train-samples 31250 \
    --load /lustre/fs1/portfolios/adlr/users/lmcafee/bert-23/checkpoints \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --data-path ${DATA_BLEND} \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
    --data-impl mmap \
    --split 98,2,0 \
    --distributed-backend nccl \
    --lr 0.0001 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --lr-decay-samples 2 \
    --lr-warmup-samples 1 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --eval-interval 2000 \
    --eval-iters 100 \
    --fp16 \
    --DDP-impl local \
    --dataloader-type cyclic \
    --no-data-sharding \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --bert-embedder-type megatron \
    --output-bert-embeddings \
    --retro-gpt-tokenizer-type GPT2BPETokenizer \
    --retro-gpt-vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/gpt2-vocab.json \
    --retro-gpt-merge-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/gpt2-merges.txt \
    --retro-gpt-seq-length 2048 \
    --retro-gpt-chunk-length 64 \
    --retro-bert-vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
    --retro-bert-tokenizer-type BertWordPieceLowerCase \
    --retro-tasks ${RETRO_TASKS} \
    --retro-index-str ${RETRO_INDEX_STR} \
    --retro-ef-search 4 \
    --retro-nprobe 64 \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-nchunks-sampled ${RETRO_INDEX_TRAIN_SAMPLES} \
    --retro-index-train-load-fraction ${RETRO_INDEX_TRAIN_LOAD_FRACTION} \
    --retro-index-add-load-fraction ${RETRO_INDEX_ADD_LOAD_FRACTION} \
    --retro-return-doc-ids \
    --retro-no-delete-index-training-embeddings \
"

# eof
