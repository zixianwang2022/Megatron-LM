#!/bin/bash

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

DIR=$(readlink -f `pwd`)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

######## Arguments. ########

. args_wiki.sh

######## Command. ########

# tools/retro/main.py
# megatron/core/models/retro/data/preprocess.py ${ARGS} \
CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
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

# eof
