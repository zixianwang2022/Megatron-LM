#!/bin/bash

set -u

# echo "SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
# NPROCS=$SLURM_TASKS_PER_NODE
# >>>
NPROCS=1
# NPROCS=8
# NPROCS=16
# NPROCS=128
# >>>

# Data blend.
. /gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/gpt3_blend.sh

# >>>>>>>>>>>>>>>>>>>>>>>
# profile_stage_stop="preprocess"
profile_stage_stop="cluster"

# tasks="clean-data"
# tasks="split-data"
# tasks="gen-rand-data"
tasks="preprocess-chunks" # "embed-preprocess"
# tasks="embed-chunks"
# tasks=train
# tasks=add
# tasks="remove-train-outputs,train"
# tasks="remove-add-outputs,add"
# tasks="remove-add-outputs"
# tasks="remove-add-outputs,verify" # "verify-index"
# tasks="verify-codes"
# tasks="verify-nbrs"
# tasks="query"
# tasks="plot-acc"
# tasks="time-hnsw"
# tasks="time-query"
# tasks="time-merge-partials"
# tasks="copy-corpus-dirty"
# tasks="nan-stats"
# tasks="bert-nan-analysis"

# ntrain=2048 ncluster=64 hnsw=4
# ntrain=131072 ncluster=128 hnsw=32
# ntrain=5000000 ncluster=100000 hnsw=32
# ntrain=15000000 ncluster=500000 hnsw=32
# ntrain=20000000 ncluster=4194304 hnsw=32
# ntrain=50000000 nadd=200000000 ncluster=4194304 hnsw=32
# ntrain=300000000 ncluster=4194304 hnsw=32
# ntrain=50000 nadd=20000000 ncluster=16384 hnsw=32
# ntrain=50000 nadd=8000000 ncluster=16384 hnsw=32
# ntrain=2500000 nadd=20000000 ncluster=262144 hnsw=32
# ntrain=2500000 nadd=100000000 ncluster=262144 hnsw=32
# ntrain=2500000 nadd=20000000 ncluster=262144 hnsw=32
# ntrain=2500000 nadd=$(($NPROCS*1000000)) ncluster=262144 hnsw=32
# ntrain=2500000 nadd=4000000 ncluster=262144 hnsw=32
# ntrain=500000 nadd=10000000 ncluster=262144 hnsw=32
# ntrain=10000000 nadd=20000000 ncluster=1048576 hnsw=32
# ntrain=3000000 nadd=100000000 ncluster=1048576 hnsw=32
# ntrain=3000000 nadd=$(($NPROCS*1000000)) ncluster=1048576 hnsw=32
# ntrain=100000000 nadd=$(($NPROCS*1000000)) ncluster=4194304 hnsw=32
ntrain=100000000 nadd=$((1*$NPROCS*1000000)) ncluster=4194304 hnsw=32

pq_dim=32
ivf_dim=256

# data_ty=corpus
# data_ty=corpus-clean
# data_ty=corpus-dirty
# data_ty=wiki
# data_ty=rand-1m
# data_ty=rand-100k

index_ty=faiss-base
# index_ty=faiss-par-add
# index_ty=faiss-decomp

# [no] bert_load_path=/home/universal-lm-data-netapp/chkpts/bert/345m_cased
# bert_load_path=/home/universal-lm-data-netapp/chkpts/bert/345M_no_rng
# >>>
# token_data_path=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/bc_rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_mmap
# token_vocab_file=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt
# +++
# data_path=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt2_indexed_dataset/sample_dataset/wikidump_10k_text_document
# data_path=${DATA_BLEND}
VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt
MERGE_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json
# <<<
# data_dir=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/data/$data_ty
index_dir=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/index
# PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-index-add
PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-preprocess

SEED=1001
EMBED_START=0
EMBED_END=100
NEIGHBOR_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/preprocess/neighbors.hdf5
OFFSET_DICT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/preprocess/offset_dict.pkl

#     --bert-load-path ${bert_load_path} \
#     --data-ty ${data_ty} \
#     --data-dir ${data_dir} \
#     --profile-stage-stop ${profile_stage_stop} \
RETRIEVAL_ARGS=" \
    --tasks ${tasks} \
    --ntrain ${ntrain} \
    --nadd ${nadd} \
    --ncluster ${ncluster} \
    --hnsw-m ${hnsw} \
    --ivf-dim ${ivf_dim} \
    --pq-m ${pq_dim} \
    --index-dir ${index_dir} \
    --index-ty ${index_ty} \

    --return_doc_ids \
    --start ${EMBED_START} \
    --end ${EMBED_END} \
    --add_offset_doc_ids \
    --offset_dict_path ${OFFSET_DICT_PATH} \
    --neighbors_path ${NEIGHBOR_PATH} \
"
#     --save $FINETUNED_PATH \
#     --load $CHECKPOINT_PATH \
#     --log-validation-ppl-to-tensorboard \
#     --tensorboard-dir ${TENSORBOARD_DIR} \
MEGATRON_ARGS=" \
    --seed ${SEED} \
    --tokenizer-typee GPT2BPETokenizer \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --data-path ${DATA_BLEND} \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --data-impl mmap \
    --split 98,2,0 \
    --distributed-backend nccl \
    --lr-warmup-samples 162761 \
    --lr-decay-style cosine \
    --lr 3.0e-4 \
    --min-lr 3.0e-5 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --checkpoint-activations \
    --log-interval 100 \
    --eval-iters 25600 \
    --eval-interval 2000 \
    --save-interval 10000 \
    --fp16 \
    --DDP-impl local \
    --finetune \
    --no-load-optim \
    --weight 0 \
"
RETRIEVAL_PREPROCESS_CMD=" \
    python -m torch.distributed.launch \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ./tools/retrieval/main.py \
    ${RETRIEVAL_ARGS} \
    ${MEGATRON_ARGS} \
"
# eof
