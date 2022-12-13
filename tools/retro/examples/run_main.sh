#!/bin/bash

set -u
unset NCCL_DEBUG

NPROCS=1
# NPROCS=2
# NPROCS=4
# NPROCS=8
# NPROCS=16

######## Data corpus. ########
# CORPUS="play"
# CORPUS="wiki"
# CORPUS="wiki-1m"
CORPUS="nih"
# CORPUS="corpus"

######## Repo. ########
REPO="retro"
# REPO="retro-wiki"
# REPO="retro-corpus"

######## Environment vars. ########
DIR=$(dirname "$0")
source ${DIR}/get_cmd.sh

# echo "RETRO_PREPROCESS_CMD = ${RETRO_PREPROCESS_CMD}"
# exit

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
