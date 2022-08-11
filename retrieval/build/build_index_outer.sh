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

# DIR=$(readlink -f `pwd`)
# DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# LOG_DIR=$DIR/logs/build
LOG_DIR="./retrieval/build/logs"
mkdir -p $LOG_DIR

run_cmd="pwd && cd $SHARE_SOURCE/megatrons/megatron-lm-retrieval-index-add && pwd && bash retrieval/build/build_index_inner.sh"

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
