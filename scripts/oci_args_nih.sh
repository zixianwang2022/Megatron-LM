#!/bin/bash

# ... /lustre/fsw/adlr/adlr-nlp/lmcafee
# ... /lustre/fs1/portfolios/adlr/users/lmcafee

# REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-retro-next-llm"
REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-retro-dedupe-sqlite"

######## task (db, index, query). ########

# RETRO_TASKS="db-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
RETRO_TASKS="query-pretraining-neighbors"

######## data. ########

if [ "0" = "0" ]; then
    RETRO_WORKDIR="/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/nih"
    DATA_BLEND="1.000 /lustre/fs1/portfolios/adlr/users/lmcafee/retro/data/MTNLG/NIHExporter_shuf_text_document"
else
    RETRO_WORKDIR="/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/nih-books2"
    DATA_BLEND="\
        0.5 /lustre/fs1/portfolios/adlr/users/lmcafee/retro/data/MTNLG/NIHExporter_shuf_text_document \
        0.5 /lustre/fs1/portfolios/adlr/users/lmcafee/retro/data/MTNLG/BookCorpus2_shuf_text_document \
    "
fi

######## index. ########

# RETRO_INDEX_STR="IVF262144_HNSW32,Flat"
# RETRO_INDEX_STR="IVF65536_HNSW8,Flat"
RETRO_INDEX_STR="OPQ32_64,IVF65536_HNSW8,PQ32"

# RETRO_INDEX_NTRAIN=10000000
RETRO_INDEX_NTRAIN=1000000
RETRO_INDEX_TRAIN_LOAD_FRACTION=0.97
RETRO_INDEX_ADD_LOAD_FRACTION=0.95

######## gpt. ########

RETRO_GPT_DATA_SPLIT="98,2,0"
RETRO_GPT_DATALOADER_TYPE=single
RETRO_GPT_TRAIN_SAMPLES=1000000
RETRO_GPT_LR_DECAY_SAMPLES=900000
RETRO_GPT_LR_WARMUP_SAMPLES=10000
RETRO_GPT_EVAL_INTERVAL=2000
RETRO_GPT_EVAL_ITERS=50
# RETRO_GPT_HIDDEN_SIZE=2048
RETRO_GPT_SEQ_LENGTH=512
RETRO_GPT_GLOBAL_BATCH_SIZE=256
RETRO_GPT_CHUNK_LENGTH=64

######## common args. ########

. ./oci_args_common.sh

# eof
