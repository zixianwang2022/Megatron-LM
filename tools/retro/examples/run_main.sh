#!/bin/bash

set -u
unset NCCL_DEBUG

# NPROCS=1 # debug.
# NPROCS=2 # query corpus.
# NPROCS=4 # query wiki.
# NPROCS=8 # index.
NPROCS=16 # index.

######## Environment vars. ########
# DIR=$(dirname "$0")
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${DIR}/get_cmd.sh

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "DIR = '$DIR'."
echo "RETRO_PREPROCESS_CMD = '$RETRO_PREPROCESS_CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

######## Command. ########
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
