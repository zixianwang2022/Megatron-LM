#!/bin/bash

set -u

# echo "SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
# NPROCS=$SLURM_TASKS_PER_NODE
# >>>
NPROCS=4
# >>>

# >>>>>>>>>>>>>>>>>>>>>>>
# profile_stage_stop="preprocess"
profile_stage_stop="cluster"

# tasks="clean-data"
# tasks="split-data"
# tasks="gen-rand-data"
tasks=train
# tasks=add
# tasks="remove-train-outputs,train"
# tasks="remove-add-outputs,add"
# tasks="remove-add-outputs"
# tasks="time-merge-partials"
# tasks="remove-add-outputs,verify" # "verify-index"
# tasks="verify-codes"
# tasks="verify-nbrs"
# tasks="query-acc"

# ntrain=2048 ncluster=64 hnsw=4
# ntrain=131072 ncluster=128 hnsw=32
# ntrain=5000000 ncluster=100000 hnsw=32
# ntrain=15000000 ncluster=500000 hnsw=32
# ntrain=20000000 ncluster=4194304 hnsw=32
# ntrain=50000000 nadd=200000000 ncluster=4194304 hnsw=32
# ntrain=300000000 ncluster=4194304 hnsw=32
# ntrain=50000 nadd=20000000 ncluster=16384 hnsw=32
ntrain=50000 nadd=8000000 ncluster=16384 hnsw=32
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

pq_dim=32
ivf_dim=256

# data_ty=corpus
data_ty=corpus-clean
# data_ty=wiki
# data_ty=rand-1m
# data_ty=rand-100k

# index_ty=faiss-base
# index_ty=faiss-decomp
index_ty=faiss-par-add

data_dir=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/data/$data_ty
index_dir=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/index
PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-index-add

# BUILD_INDEX_CMD=" \
#     ${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-index-add/retrieval/build/build_index.py \
# --data-ty ${data_ty} \
BUILD_INDEX_CMD=" \
    ./tools/retrieval/main.py \
    --tasks ${tasks} \
    --ntrain ${ntrain} \
    --nadd ${nadd} \
    --ncluster ${ncluster} \
    --hnsw-m ${hnsw} \
    --ivf-dim ${ivf_dim} \
    --pq-m ${pq_dim} \
    --data-dir ${data_dir} \
    --index-dir ${data_dir} \
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
