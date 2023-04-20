#!/bin/bash

# REPO_DIR=/home/lmcafee/src/megatrons/megatron-lm-retro-next-llm
REPO_DIR=/home/lmcafee/src/megatrons/megatron-lm-retro-dedupe-sqlite

# RETRO_WORKDIR="/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/reddit-plus-mt-lower"
RETRO_WORKDIR="/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/next-llm"

# RETRO_TASKS="db-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
RETRO_TASKS="query-pretraining-neighbors"

######## data. ########

. /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/lawrence_blend_oci.sh

######## index. ########

# RETRO_INDEX_STR="IVF4096_HNSW4,Flat"
# RETRO_INDEX_STR="OPQ32_256,IVF4194304_HNSW32,PQ32"
RETRO_INDEX_STR="OPQ64_128,IVF4194304_HNSW32,PQ64"
# RETRO_INDEX_STR="OPQ128_64,IVF4194304_HNSW32,PQ128"

RETRO_INDEX_NTRAIN=600000000

# RETRO_INDEX_TRAIN_LOAD_FRACTION=0.33
# RETRO_INDEX_TRAIN_LOAD_FRACTION=0.42
# RETRO_INDEX_TRAIN_LOAD_FRACTION=0.5 # *
RETRO_INDEX_TRAIN_LOAD_FRACTION=0.66667

RETRO_INDEX_ADD_LOAD_FRACTION=1.0
# RETRO_INDEX_ADD_LOAD_FRACTION=0.9
# RETRO_INDEX_ADD_LOAD_FRACTION=0.875
# RETRO_INDEX_ADD_LOAD_FRACTION=0.85

######## gpt. ########

RETRO_GPT_DATA_SPLIT="98,2,0"
RETRO_GPT_DATALOADER_TYPE=single
RETRO_GPT_TRAIN_SAMPLES=268554688
RETRO_GPT_LR_DECAY_SAMPLES=255126953
RETRO_GPT_LR_WARMUP_SAMPLES=122071
RETRO_GPT_EVAL_INTERVAL=2000
RETRO_GPT_EVAL_ITERS=50
# RETRO_GPT_HIDDEN_SIZE=2048
RETRO_GPT_SEQ_LENGTH=4096
RETRO_GPT_GLOBAL_BATCH_SIZE=256
RETRO_GPT_CHUNK_LENGTH=64

######## common args. ########

. ./oci_args_common.sh

# eof
