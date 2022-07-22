#!/bin/bash

#SBATCH -p batch --nodes=2 --gres=gpu:8 -A gpu_adlr_nlp -t 0:10:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retrieval

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOG_DIR="./lawrence/sandbox/rapids/logs"
mkdir -p $LOG_DIR

run_cmd="cd $DIR && bash lawrence/sandbox/rapids/main_inner.sh"

if [[ $HOSTNAME == *"rno"* ]]; then
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee,/gpfs/fs1/projects/gpu_adlr/datasets/boxinw:/gpfs/fs1/projects/gpu_adlr/datasets/boxinw"

elif [[ $HOSTNAME == *"luna-"* ]]; then
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/lustre/fsw/adlr:/lustre/fsw/adlr"

elif [[ $HOSTNAME == *"ip-"* ]]; then
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/mnt/fsx-outputs-chipdesign/lmcafee:/mnt/fsx-outputs-chipdesign/lmcafee"

else
    echo "error ... specialize for host '$HOSTNAME'."
    exit 1
fi

srun -l \
     --container-image "nvcr.io#nvidia/pytorch:22.04-py3" \
     --container-mounts $CONTAINER_MOUNTS \
     --output=$LOG_DIR/%j.log \
     sh -c "${run_cmd}"

set +x

# eof
