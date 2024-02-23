#!/bin/bash

set -u
unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPROCS=1 # 8

######## Arguments. ########

# . args_tiny_wiki.sh
. args_wiki.sh
# . args_nextlm.sh

######## Command. ########

# tools/retro/main.py
# megatron/core/models/retro/data/preprocess.py ${ARGS} \
# export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:${PYTHONPATH} && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    tools/retro/preprocess_data.py ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.
