#!/bin/bash

#SBATCH -p batch_block1,batch_block2
#SBATCH --nodes=4
#SBATCH -A adlr
#SBATCH -t 2:00:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:retro-next-llm
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

DIR=$(readlink -f `pwd`)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

######## Arguments. ########
. oci_args.sh
# . oci_args_nih.sh

######## Command. ########
CMD="export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && python -u ${REPO_DIR}/tools/retro/main.py ${ARGS}"
# MOUNTS="/home/lmcafee:/home/lmcafee,/lustre/fsw/adlr/adlr-nlp/lmcafee:/lustre/fsw/adlr/adlr-nlp/lmcafee,/lustre/fsw/adlr/adlr-nlp/data:/lustre/fsw/adlr/adlr-nlp/data"
# MOUNTS="/home/lmcafee:/home/lmcafee,/lustre/fsw/adlr/adlr-nlp/lmcafee:/lustre/fsw/adlr/adlr-nlp/lmcafee,/lustre/fsw/adlr/adlr-nlp/lmcafee:/lustre/fsw/adlr/adlr-nlp/lmcafee"
MOUNTS="/home/lmcafee:/home/lmcafee,/lustre/fs1/portfolios/adlr/users/lmcafee:/lustre/fs1/portfolios/adlr/users/lmcafee"

# export RX_QUEUE_LEN=8192
# export IB_RX_QUEUE_LEN=8192
# export NCCL_IB_TIMEOUT=16
# export NCCL_IB_SL=0
# export NCCL_IB_TC=41
# export NCCL_IGNORE_CPU_AFFINITY=1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_QPS_PER_CONNECTION=4

# IMAGE="nvcr.io#nvidia/pytorch:22.04-py3"
# IMAGE=gitlab-master.nvidia.com/lmcafee/sandbox-cluster/retro-process
IMAGE=gitlab-master.nvidia.com/lmcafee/sandbox-cluster/retro-process-22.12
# IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch_flash_att:22.09-py3"
# IMAGE=http://gitlab-master.nvidia.com/adlr/megatron-lm/pytorch_flash_att:22.12-py3-sentencepiece
srun -l \
     --container-image ${IMAGE} \
     --container-mounts ${MOUNTS} \
     --output=$DIR/logs/"%j_${RETRO_TASKS}.log" \
     sh -c "${CMD}"

# eof
