#!/bin/bash

#SBATCH -p batch --nodes=2 --gres=gpu:8 -A gpu_adlr_nlp -t 1:00:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retrieval --export=NPROCS=4

# --ntasks-per-node=4
NNODES=2
NPROCS=4

# >>>
# [selene] SBATCH -p luna -A adlr -t 4:00:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=1 --job-name=adlr-nlp-dev:retrieval
# [draco-rno] SBATCH -p batch_dgx2h_m2 --gres=gpu:16 -A gpu_adlr_nlp -t 8:00:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retrieval
# [draco-rno] *batch_dgx2h_m2, batch_short_dgx2h_m2
# <<<

# NAME="lmcafee-retrieval"

# DIR=$(readlink -f `pwd`)
# DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# LOG_DIR=$DIR/logs/build
# LOG_DIR="/home/lmcafee-src/megatrons/megatron-lm-boxin/lawrence/logs/build"
# LOG_DIR="./lawrence/logs/build"
# LOG_DIR="./lawrence/build/logs"
LOG_DIR="./retrieval/build/logs"
mkdir -p $LOG_DIR

# export NNODES=1
# export NPROCS=4

# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # task=train
# task=add
# # task=query

# profile_stage_stop=opq
# # profile_stage_stop=ivf
# # profile_stage_stop=pqs

# # ntrain=10000
# # ntrain=100000
# # ntrain=200000
# # ntrain=500000
# # ntrain=1000000
# # ntrain=20000000
# ntrain=50000000
# # ntrain=100000000 # *
# # ntrain=200000000
# # ntrain=300000000
# # ntrain=400000000

# # ncluster=128
# # ncluster=512
# # ncluster=4096
# # ncluster=8192
# # ncluster=16384
# # ncluster=262144
# ncluster=4194304 # *

# # data_ty=wiki
# data_ty=corpus

# # index_ty=faiss-mono
# index_ty=faiss-decomp
# # index_ty=cuml
# # index_ty=distrib

# # index_ty=fancy # simple
# # index_str="Flat"
# # index_str="IVF${ncluster}_HNSW32,Flat"

# # index_str="OPQ64_64,IVF${ncluster}_HNSW32,PQ64"
# # index_str="OPQ64_128,IVF${ncluster}_HNSW32,PQ64" # boxin base
# # index_str="OPQ64_256,IVF${ncluster}_HNSW32,PQ64"
# # index_str="OPQ64_512,IVF${ncluster}_HNSW32,PQ64"

# # index_str="OPQ32_64,IVF${ncluster}_HNSW32,PQ32"
# # index_str="OPQ32_128,IVF${ncluster}_HNSW32,PQ32"
# index_str="OPQ32_256,IVF${ncluster}_HNSW32,PQ32" # lawrence base
# # index_str="OPQ32_512,IVF${ncluster}_HNSW32,PQ32"

# # index_str="OPQ16_8,IVF${ncluster}_HNSW32,PQ16"
# # index_str="OPQ16_16,IVF${ncluster}_HNSW32,PQ16"
# # index_str="OPQ16_32,IVF${ncluster}_HNSW32,PQ16"
# # index_str="OPQ16_64,IVF${ncluster}_HNSW32,PQ16"
# # index_str="OPQ16_128,IVF${ncluster}_HNSW32,PQ16"
# # index_str="OPQ16_256,IVF${ncluster}_HNSW32,PQ16"
# # index_str="OPQ16_512,IVF${ncluster}_HNSW32,PQ16"
# # <<<

# # >>>
# cmd="python -m lawrence.build.build_index \
#             --task ${task} \
#             --data-ty corpus \
#             --ntrain ${ntrain} \
#             --index-ty ${index_ty} \
#             --index-str ${index_str} \
#             --profile-stage-stop ${profile_stage_stop} \
#             --profile-single-encoder \
# "
# # cmd="ls -alh"
# # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# run_cmd="pwd && cd $DIR && pwd && bash lawrence/build/build_index_inner.sh $task $data_ty $index_ty $index_str $ntrain"
# run_cmd="pwd && cd $DIR && pwd && bash lawrence/build/build_index_inner.sh '$cmd'"
# run_cmd="pwd && cd $DIR && pwd && bash retrieval/build/build_index_inner.sh"
run_cmd="pwd && cd $SHARE_SOURCE/megatrons/megatron-lm-retrieval-index-add && pwd && bash retrieval/build/build_index_inner.sh"

# >>>
# echo "PWD = $(pwd)"
# echo "SHARE_SOURCE = $SHARE_SOURCE"
# echo "RUN_CMD = $run_cmd"
# exit 0
# <<<

if [[ $HOSTNAME == *"rno"* ]]; then
    IMAGE=nvcr.io#nvidia/pytorch:22.04-py3
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee,/gpfs/fs1/projects/gpu_adlr/datasets/boxinw:/gpfs/fs1/projects/gpu_adlr/datasets/boxinw"

elif [[ $HOSTNAME == *"luna-"* ]]; then
    IMAGE=nvcr.io#nvidia/pytorch:22.04-py3
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/lustre/fsw/adlr:/lustre/fsw/adlr"

elif [[ $HOSTNAME == *"ip-"* ]]; then
    IMAGE=gitlab-master.nvidia.com/adlr/adlr-utils/aws/pytorch-efa:pytorch-21.12
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/mnt/fsx-outputs-chipdesign/lmcafee:/mnt/fsx-outputs-chipdesign/lmcafee"

else
    echo "error ... specialize for host '$HOSTNAME'."
    exit 1
fi

# --output=$LOG_DIR/%j__${task}__${data_ty}__${index_ty}__${index_str}__t${ntrain}.log sh -c "${run_cmd}"
# --output=$LOG_DIR/%j__${index_ty}__${index_str}__${ntrain}__${task}__${profile_stage_stop}__${data_ty}.log sh -c "${run_cmd}"
srun -l \
     --container-image $IMAGE \
     --container-mounts $CONTAINER_MOUNTS \
     --output=$LOG_DIR/%j__n${NNODES}__p${NPROCS}.log sh -c "${run_cmd}"

set +x

# eof
