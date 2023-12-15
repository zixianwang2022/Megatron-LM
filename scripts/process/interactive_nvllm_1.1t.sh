#!/bin/bash

#SBATCH -p batch_block1,batch_block2
#SBATCH --nodes=1
#SBATCH -A adlr
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:retro-next-llm
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton







set -u
unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

# >>>
NPROCS=8
# <<<

######## tasks. ########

RETRO_TASKS="db-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
# RETRO_TASKS="query-neighbors"

# RETRO_TASK_VALIDATE=""
# RETRO_TASK_VALIDATE=1
RETRO_TASK_VALIDATE=0.1






# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# customize / begin.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# REPO_DIR="/home/lmcafee/src/megatrons/megatron-lm-retro-dedupe-sqlite"
# RETRO_WORKDIR="/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/next-llm"
REPO_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/retro-mcore-data"
RETRO_PROJECT_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/projects/nvllm-1.1t-core"

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# customize / end.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<






# **** warning: shouldn't need to edit below. ****

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

DIR=$(readlink -f `pwd`)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

######## data. ########
# . /lustre/fsw/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/lawrence_blend_oci.sh
. ${RETRO_PROJECT_DIR}/data/blend.sh

######## index. ########

RETRO_INDEX_STR="OPQ64_128,IVF4194304_HNSW32,PQ64"
RETRO_INDEX_NTRAIN=600000000
RETRO_INDEX_TRAIN_LOAD_FRACTION=0.66667
RETRO_INDEX_ADD_LOAD_FRACTION=1.0

######## gpt. ########

RETRO_GPT_SEED=1234
RETRO_GPT_SPLIT="98,2,0"
RETRO_GPT_DATA_PATH=${DATA_BLEND}
RETRO_GPT_DATALOADER_TYPE=single
RETRO_GPT_TRAIN_SAMPLES=25000000
RETRO_GPT_EVAL_INTERVAL=2000
RETRO_GPT_EVAL_ITERS=32
RETRO_GPT_LR_DECAY_SAMPLES=2
RETRO_GPT_LR_WARMUP_SAMPLES=1
# RETRO_GPT_HIDDEN_SIZE=2048
RETRO_GPT_SEQ_LENGTH=4096
RETRO_GPT_GLOBAL_BATCH_SIZE=768
RETRO_GPT_CHUNK_LENGTH=64

######## query. ########

RETRO_QUERY_NUM_NEIGHBORS_QUERY=200
RETRO_QUERY_NUM_NEIGHBORS_SAVE=20
RETRO_QUERY_EF_SEARCH=32
RETRO_QUERY_NPROBE=4096

######## args. ########

#     --data-impl mmap \
#     --DDP-impl local \
#     --retro-return-doc-ids \
#     --retro-index-no-delete-training-embeddings \
#     --retro-index-no-delete-added-codes \
#     --data-path ${DATA_BLEND} \
# --retro-gpt-tokenizer-model /lustre/fsw/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
# --retro-bert-vocab-file /lustre/fsw/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
# --vocab-file /lustre/fsw/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
# --load /lustre/fsw/portfolios/adlr/users/lmcafee/bert-23/checkpoints \
ARGS=" \
    --distributed-timeout-minutes 600 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 1 \
    --global-batch-size ${RETRO_GPT_GLOBAL_BATCH_SIZE} \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load ${RETRO_PROJECT_DIR}/checkpoints/bert \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --data-path [null] \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file ${RETRO_PROJECT_DIR}/tokenizer/bert-large-uncased-vocab.txt \
    --split ${RETRO_GPT_SPLIT} \
    --distributed-backend nccl \
    --lr 0.0001 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --train-samples ${RETRO_GPT_TRAIN_SAMPLES} \
    --lr-decay-samples ${RETRO_GPT_LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${RETRO_GPT_LR_WARMUP_SAMPLES} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --eval-interval ${RETRO_GPT_EVAL_INTERVAL} \
    --eval-iters ${RETRO_GPT_EVAL_ITERS} \
    --fp16 \
    --dataloader-type ${RETRO_GPT_DATALOADER_TYPE} \
    --no-data-sharding \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --bert-embedder-type megatron \
    --output-bert-embeddings \
    \
    --retro-project-dir ${RETRO_PROJECT_DIR} \
    --retro-tasks ${RETRO_TASKS} \
    --retro-bert-vocab-file tokenizer/bert-large-uncased-vocab.txt \
    --retro-bert-tokenizer-type BertWordPieceLowerCase \
    \
    --retro-gpt-seed ${RETRO_GPT_SEED} \
    --retro-gpt-tokenizer-type GPTSentencePieceTokenizer \
    --retro-gpt-tokenizer-model tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --retro-gpt-seq-length ${RETRO_GPT_SEQ_LENGTH} \
    --retro-gpt-chunk-length ${RETRO_GPT_CHUNK_LENGTH} \
    --retro-gpt-global-batch-size ${RETRO_GPT_GLOBAL_BATCH_SIZE} \
    --retro-gpt-eval-interval ${RETRO_GPT_EVAL_INTERVAL} \
    --retro-gpt-eval-iters ${RETRO_GPT_EVAL_ITERS} \
    --retro-gpt-split ${RETRO_GPT_SPLIT} \
    --retro-gpt-data-path ${RETRO_GPT_DATA_PATH} \
    --retro-gpt-train-samples ${RETRO_GPT_TRAIN_SAMPLES} \
    \
    --retro-index-str ${RETRO_INDEX_STR} \
    --retro-index-ntrain ${RETRO_INDEX_NTRAIN} \
    --retro-index-train-load-fraction ${RETRO_INDEX_TRAIN_LOAD_FRACTION} \
    --retro-index-add-load-fraction ${RETRO_INDEX_ADD_LOAD_FRACTION} \
    --no-retro-index-delete-training-embeddings \
    --no-retro-index-delete-added-codes \
    \
    --retro-query-num-neighbors-query ${RETRO_QUERY_NUM_NEIGHBORS_QUERY} \
    --retro-query-num-neighbors-save ${RETRO_QUERY_NUM_NEIGHBORS_SAVE} \
    --retro-query-ef-search ${RETRO_QUERY_EF_SEARCH} \
    --retro-query-nprobe ${RETRO_QUERY_NPROBE} \
"
if [ "${RETRO_TASK_VALIDATE}" != "" ]; then
    ARGS+=" --retro-task-validate ${RETRO_TASK_VALIDATE}"
fi

# ######## srun. ########

# CMD="export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && python -u ${REPO_DIR}/tools/retro/main.py ${ARGS}"
# MOUNTS="/home/lmcafee:/home/lmcafee,/lustre/fs1/portfolios/adlr/users/lmcafee:/lustre/fs1/portfolios/adlr/users/lmcafee"
# IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-22.12"

# srun -l \
#      --container-image ${IMAGE} \
#      --container-mounts ${MOUNTS} \
#      --output=$DIR/logs/"%j_${RETRO_TASKS}.log" \
#      sh -c "${CMD}"

######## Command. ########

# tools/retro/main.py
# megatron/core/models/retro/data/preprocess.py ${ARGS} \
CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    tools/retro/preprocess_data.py ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof
