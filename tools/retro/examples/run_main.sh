#!/bin/bash

set -u

unset NCCL_DEBUG

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
source $PARENT_PATH/get_cmd.sh

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "PARENT_PATH = '$PARENT_PATH'."
echo "RETRO_PREPROCESS_CMD = '$RETRO_PREPROCESS_CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

# NPROCS=1
# NPROCS=2
# NPROCS=4
# NPROCS=8
NPROCS=16

REPO="retro"
# REPO="retro-wiki"
# REPO="retro-corpus"

FULL_CMD="\
    pwd && cd $SHARE_SOURCE/megatrons/megatron-lm-${REPO} && pwd && \
    export PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-${REPO}&&\
    python -m torch.distributed.launch \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    $RETRO_PREPROCESS_CMD \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "FULL_CMD = '$FULL_CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $FULL_CMD
