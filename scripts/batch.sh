#!/bin/bash

#SBATCH -p batch_block1,batch_block2
#SBATCH --nodes=1
#SBATCH -A adlr
#SBATCH -t 0:30:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:flash-off
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton

FLASH=off

######## Arguments. ########

. args.sh

######## Command. ########

# DIR=$(readlink -f `pwd`)
# DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# LOG_DIR=$DIR/logs
# mkdir -p $LOG_DIR
LOG_DIR=$CHECKPOINT_DIR/logs
mkdir -p $LOG_DIR

# export PYTHONPATH=$PYTHONPATH:${REPO_DIR}:/home/lmcafee/src && \
CMD=" \
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
    python -u ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo $CMD
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

if [ "${FLASH}" = "off" ]; then
    IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-22.12"
elif [ "${FLASH}" = "0" ]; then
    IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-22.12"
elif [ "${FLASH}" = "1" ]; then
    IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-22.12-flash2"
else
    echo "use new flash?"
    exit 1
fi
# IMAGE="/lustre/fsw/adlr/adlr-nlp/images/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels.sqsh"

# MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/lustre/fsw/adlr/adlr-nlp/lmcafee:/lustre/fsw/adlr/adlr-nlp/lmcafee,/lustre/fs1/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t:/lustre/fs1/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t"
MOUNTS="/home/lmcafee/src:/home/lmcafee/src"
# MOUNTS="${MOUNTS},/lustre/fsw/adlr/adlr-nlp/lmcafee:/lustre/fsw/adlr/adlr-nlp/lmcafee"
MOUNTS="${MOUNTS},/lustre/fs1/portfolios/adlr/users/lmcafee:/lustre/fs1/portfolios/adlr/users/lmcafee"
MOUNTS="${MOUNTS},/lustre/fs1/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t:/lustre/fs1/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t"

srun -l \
     --container-image $IMAGE \
     --container-mounts $MOUNTS \
     --output=$LOG_DIR/"%j.log" \
     sh -c "${CMD}"

# eof.
