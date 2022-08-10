#!/bin/bash

#SBATCH -p batch --gres=gpu:8 -A gpu_adlr_nlp -t 4:00:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retrieval --dependency=singleton

# DIR=$(readlink -f `pwd`)
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# LOG_DIR=$DIR/logs/build
LOG_DIR="/home/lmcafee/src/megatrons/megatron-lm-retrieval-index-add/retrieval/transfer/logs"
mkdir -p $LOG_DIR

run_cmd="pwd && cd $DIR && pwd && bash retrieval/transfer/transfer_inner.sh"
# CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee,/gpfs/fs1/projects/gpu_adlr/datasets/boxinw:/gpfs/fs1/projects/gpu_adlr/datasets/boxinw"
CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/mnt/fsx-outputs-chipdesign/lmcafee:/mnt/fsx-outputs-chipdesign/lmcafee"

srun -l \
     --container-image "nvcr.io#nvidia/pytorch:22.04-py3" \
     --container-mounts $CONTAINER_MOUNTS \
     --output=$LOG_DIR/%j.log sh -c "${run_cmd}"

set +x

# eof
