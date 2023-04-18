#!/bin/bash

# ... /lustre/fsw/adlr/adlr-nlp/lmcafee
# ... /lustre/fs1/portfolios/adlr/users/lmcafee

# REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-retro-next-llm"
REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-retro-dedupe-sqlite"

RETRO_TASKS="db-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
# RETRO_TASKS="pretraining-query-neighbors"

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

######## common args. ########

. ./oci_args_common.sh

# eof
