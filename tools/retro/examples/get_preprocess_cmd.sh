#!/bin/bash

"""
Build preprocessing command for Retro.
"""

set -u
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

######## Environment variables. ########
# Required environment variables:
# - RETRO_WORKDIRS : Root directory that contains independent Retro projects
#     (e.g., for training on different datasets or blends). Each project sub-
#     directory will contain a complete set of preprocessed data, including
#     the retrieval database, search index, and pretraining neighbors.
# - BLEND_SCRIPT_DIR : Directory containing blended dataset definition files.
#     Loaded script will be '${BLEND_SCRIPT_DIR}/data_blend_${CORPUS}.sh'.
# - GPT_VOCAB_FILE : GPT vocab file.
# - GPT_MERGE_FILE : GPT merge file.
# - GPT_TOKENIZER : GPT tokenizer type (e.g., GPT2BPETokenizer)
# - BERT_LOAD_PATH : Bert checkpoint directory.
# - BERT_VOCAB_FILE : Bert vocab file.
# - BERT_TOKENIZER : Bert tokenizer type (e.g., BertWordPieceLowerCase,
#     BertWordPieceCase).

# *Note*: The variables above can be set however a user would like. In our
# setup, we use another bash script (location defined in $RETRO_ENV_VARS) that
# sets all the environment variables at once.
. $RETRO_ENV_VARS

################ Data corpus. ################
CORPUS="wiki"
# CORPUS="wiki-tiny"
# CORPUS="corpus"

. ${DIR}/get_corpus_config.sh

################ Repo. ################
REPO="retro"

################ Data blend. ################
. ${BLEND_SCRIPT_DIR}/data_blend_${CORPUS}.sh
DATA_PATH=${DATA_BLEND}

################ Retro setup. ################
RETRO_WORKDIR=${RETRO_WORKDIRS}/${CORPUS}
RETRO_GPT_SEQ_LENGTH=2048
RETRO_GPT_CHUNK_LENGTH=64
RETRO_GPT_MICRO_BATCH_SIZE=1 # *8
RETRO_GPT_GLOBAL_BATCH_SIZE=256
RETRO_NCHUNKS_SAMPLED=300000000

################ Retro tasks. ################
# The '--retro-tasks' argument is a comma-separated list of tasks to run, in
# sequential order. For a quick start, simply set this to 'build' to run the
# entire preprocessing pipeline. For finer control, you may specify the list of
# tasks to run. This is desirable for tuning computational resources. For
# example, training the search index is relatively fast and utilizes GPUs,
# while querying the search index is relatively slow, CPU-only, and memory
# intensive (i.e., multiple populated search indexes are loaded simultaneously).

# *Note* : Once the task(s) below have been completed -- by running either
#    1) 'build', or 2) the sequential combination of 'db-build', 'index-build',
#    and 'pretraining-query-neighbors' -- we are ready to pretrain Retro by
#    calling pretrain_retro.py.

# ---- Option #1 : Run entire pipeline. ----

RETRO_TASKS="build"

# ---- Option #2 : Run specific stages. ----
# *Note*: Run the following stages in the given order. Optionally, tune your
#   cluster setup for each stage, as described above.

# RETRO_TASKS="db-build" # ....................... run 1st
# RETRO_TASKS="index-build" # .................... run 2nd
# RETRO_TASKS="pretraining-query-neighbors" # .... run 3rd

################ Megatron args. ################
MEGATRON_ARGS=" \
    --seed 1234 \
    --distributed-timeout-minutes 600 \
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
    --exit-on-missing-checkpoint \
    --no-load-optim \
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
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
"

################ Retro args. ################
RETRO_ARGS=" \
    --bert-embedder-type ${BERT_EMBEDDER_TYPE} \
    --output-bert-embeddings \
    \
    --retro-gpt-vocab-file ${GPT_VOCAB_FILE} \
    --retro-gpt-merge-file ${GPT_MERGE_FILE} \
    --retro-gpt-tokenizer-type ${GPT_TOKENIZER} \
    --retro-gpt-seq-length ${RETRO_GPT_SEQ_LENGTH} \
    --retro-gpt-chunk-length ${RETRO_GPT_CHUNK_LENGTH} \
    --retro-bert-vocab-file ${BERT_VOCAB_FILE} \
    --retro-bert-tokenizer-type ${BERT_TOKENIZER} \
    \
    --retro-tasks ${RETRO_TASKS} \
    --retro-index-str ${RETRO_INDEX_STR} \
    --retro-ef-search ${RETRO_EF_SEARCH} \
    --retro-nprobe ${RETRO_NPROBE} \
    \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-nchunks-sampled ${RETRO_NCHUNKS_SAMPLED} \
    \
    --retro-return-doc-ids \
"

################ Command. ################
RETRO_PREPROCESS_CMD=" \
    ./tools/retro/main.py \
    ${MEGATRON_ARGS} \
    ${RETRO_ARGS} \
"
