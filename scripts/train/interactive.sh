#!/bin/bash

set -u
unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

######## Setup. ########

if [ "$#" != 2 ]; then
    echo "expected 2 args, found ${#}."
    exit 1
fi
USE_CORE=$1
ADD_RETRIEVER=$2
NWORKERS=8 # 32

######## Arguments. ########

REPO_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/retro-mcore-data"

. args_wiki.sh

LOG_INTERVAL=1 EXIT_INTERVAL=10
# LOG_INTERVAL=10 EXIT_INTERVAL=100
ARGS+=" \
    --log-interval ${LOG_INTERVAL} \
    --exit-interval ${EXIT_INTERVAL} \
"

######## Command. ########

NPROCS=1 # 8
CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.
