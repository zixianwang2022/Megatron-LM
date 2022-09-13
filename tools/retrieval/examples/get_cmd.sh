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
data_ty=corpus-dirty
# data_ty=wiki
# data_ty=rand-1m
# data_ty=rand-100k

index_ty=faiss-base
# index_ty=faiss-par-add
# index_ty=faiss-decomp

# bert_load_path=/home/universal-lm-data-netapp/chkpts/bert/345m_cased
bert_load_path=/home/universal-lm-data-netapp/chkpts/bert/345M_no_rng
# >>>
# token_data_path=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/bc_rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_mmap
# token_vocab_file=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt
# +++
token_data_path=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt2_indexed_dataset/sample_dataset/wikidump_10k_text_document
token_vocab_file=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt
token_merge_file=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json
# <<<
data_dir=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/data/$data_ty
index_dir=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/index
# PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-index-add
PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-preprocess

# BUILD_INDEX_CMD=" \
#     ${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-index-add/retrieval/build/build_index.py \
BUILD_INDEX_CMD=" \
    ./tools/retrieval/main.py \
    --tasks ${tasks} \
    --ntrain ${ntrain} \
    --nadd ${nadd} \
    --ncluster ${ncluster} \
    --hnsw-m ${hnsw} \
    --ivf-dim ${ivf_dim} \
    --pq-m ${pq_dim} \
    --bert-load-path ${bert_load_path} \
    --token-data-path ${token_data_path} \
    --token-vocab-file ${token_vocab_file} \
    --data-ty ${data_ty} \
    --data-dir ${data_dir} \
    --index-dir ${index_dir} \
    --index-ty ${index_ty} \
    --profile-stage-stop ${profile_stage_stop} \
"
if [ "0" -eq "1" ]; then
    BUILD_INDEX_CMD="python -u $BUILD_INDEX_CMD"
else
    BUILD_INDEX_CMD=" \
    python -m torch.distributed.launch \
        --nproc_per_node ${NPROCS} \
        --nnodes 1 \
        --node_rank ${NODE_RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port 6000 \
        $BUILD_INDEX_CMD \
    "
fi

# eof
