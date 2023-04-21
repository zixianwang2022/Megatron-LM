#!/bin/bash

######## Arguments. ########
# . oci_args.sh
. oci_args_nih.sh
# . oci_args_wiki-tiny.sh

######## Command. ########
unset NCCL_DEBUG
NPROCS=1
# python -m torch.distributed.launch \
# torchrun \
FULL_CMD="\
    pwd && cd ${REPO_DIR} && pwd && \
    export PYTHONPATH=$PYTHONPATH:${REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    tools/retro/main.py ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "FULL_CMD = '$FULL_CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $FULL_CMD
