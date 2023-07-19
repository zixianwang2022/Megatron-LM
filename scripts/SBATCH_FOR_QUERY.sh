#!/bin/bash

#SBATCH -p batch_block1,batch_block2
#SBATCH --nodes=16
#SBATCH -A adlr
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:retro-next-llm
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton






# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# customize / begin.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

REPO_DIR="/home/boxinw/megatron-lm-pretrain"
RETRO_GPT_TRAIN_SAMPLES=25000000
RETRO_GPT_EVAL_INTERVAL=1260
RETRO_GPT_EVAL_ITERS=32
RETRO_GPT_GLOBAL_BATCH_SIZE=768

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

######## retro. ########

RETRO_WORKDIR="/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/next-llm"
RETRO_TASKS="query-pretraining-neighbors"

######## data. ########
. /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/lawrence_blend_oci.sh

######## index. ########

RETRO_INDEX_STR="OPQ64_128,IVF4194304_HNSW32,PQ64"
RETRO_INDEX_NTRAIN=600000000
RETRO_INDEX_TRAIN_LOAD_FRACTION=0.66667
RETRO_INDEX_ADD_LOAD_FRACTION=1.0

######## gpt. ########

RETRO_GPT_DATA_SPLIT="98,2,0"
RETRO_GPT_DATALOADER_TYPE=single
RETRO_GPT_LR_DECAY_SAMPLES=0
RETRO_GPT_LR_WARMUP_SAMPLES=0
# RETRO_GPT_HIDDEN_SIZE=2048
RETRO_GPT_SEQ_LENGTH=4096
RETRO_GPT_CHUNK_LENGTH=64

######## query. ########

RETRO_QUERY_NUM_NEIGHBORS_QUERY=200
RETRO_QUERY_NUM_NEIGHBORS_SAVE=20
RETRO_QUERY_EF_SEARCH=32
RETRO_QUERY_NPROBE=4096

######## args. ########

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
    --load /lustre/fs1/portfolios/adlr/users/lmcafee/bert-23/checkpoints \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --data-path ${DATA_BLEND} \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
    --data-impl mmap \
    --split ${RETRO_GPT_DATA_SPLIT} \
    --split-constraint 99,1,0 \
    --split-constraint 98,2,0 \
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
    --DDP-impl local \
    --dataloader-type ${RETRO_GPT_DATALOADER_TYPE} \
    --no-data-sharding \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --bert-embedder-type megatron \
    --output-bert-embeddings \
    \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-tasks ${RETRO_TASKS} \
    --retro-return-doc-ids \
    --retro-bert-vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
    --retro-bert-tokenizer-type BertWordPieceLowerCase \
    --retro-gpt-tokenizer-type GPTSentencePieceTokenizer \
    --retro-gpt-tokenizer-model /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --retro-gpt-seq-length ${RETRO_GPT_SEQ_LENGTH} \
    --retro-gpt-global-batch-size ${RETRO_GPT_GLOBAL_BATCH_SIZE} \
    --retro-gpt-chunk-length ${RETRO_GPT_CHUNK_LENGTH} \
    --retro-index-str ${RETRO_INDEX_STR} \
    --retro-index-ntrain ${RETRO_INDEX_NTRAIN} \
    --retro-index-train-load-fraction ${RETRO_INDEX_TRAIN_LOAD_FRACTION} \
    --retro-index-add-load-fraction ${RETRO_INDEX_ADD_LOAD_FRACTION} \
    --retro-index-no-delete-training-embeddings \
    --retro-index-no-delete-added-codes \
    --retro-query-num-neighbors-query ${RETRO_QUERY_NUM_NEIGHBORS_QUERY} \
    --retro-query-num-neighbors-save ${RETRO_QUERY_NUM_NEIGHBORS_SAVE} \
    --retro-query-ef-search ${RETRO_QUERY_EF_SEARCH} \
    --retro-query-nprobe ${RETRO_QUERY_NPROBE} \
"

######## srun. ########

CMD="export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && python -u ${REPO_DIR}/tools/retro/main.py ${ARGS}"
MOUNTS="/home/boxinw:/home/boxinw,/home/lmcafee:/home/lmcafee,/lustre/fs1/portfolios/adlr/"
IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-22.12"

srun -l \
     --container-image ${IMAGE} \
     --container-mounts ${MOUNTS} \
     --output=$DIR/logs/"%j_${RETRO_TASKS}.log" \
     sh -c "${CMD}"

# eof
