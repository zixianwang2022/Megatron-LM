#!/bin/bash

set -u

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
source $PARENT_PATH/get_cmd.sh

# >>>
# ... now required in image
# if [ "0" -eq "1" ]; then
#     pip install h5py
#     conda install -c conda-forge -y faiss-gpu
# fi
# <<<

unset NCCL_DEBUG
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "PARENT_PATH = '$PARENT_PATH'."
echo "RETRO_PREPROCESS_CMD = '$RETRO_PREPROCESS_CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
# eval $RETRO_PREPROCESS_CMD

NPROCS=1
# NPROCS=2
# NPROCS=4
# NPROCS=8
# NPROCS=16
# NPROCS=128

# REPO="retro-preprocess-corpus"
# REPO="retro-preprocess-play"
# REPO="retro-process"
REPO="retro"

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
# eval "\
#     pwd && cd $SHARE_SOURCE/megatrons/megatron-lm-${REPO} && pwd && \
#     export PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-${REPO}&&\
#     $RETRO_PREPROCESS_CMD \
# "
