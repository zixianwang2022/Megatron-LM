#!/bin/bash

set -u

if [ "$#" != "3" ]; then
    echo "expected 3 args, found $#."
    exit 1
fi

######## Arguments. ########

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXTRA_ARGS=""
. $DIR/../args_gen.sh "$@"

SCRIPT_DIR="/lustre/fs3/portfolios/adlr/users/lmcafee/llama/2/src/megatron-lm-llama2-loader/scripts/mmlu"
TASK_OPTIONS=" \
  --data_dir=${SCRIPT_DIR}/data \
  --save_dir=${SCRIPT_DIR}/save \
  ${ARGS} \
"

######## Command. ########

# cd ${MEGATRON_REPO_DIR} && \
CMD="\
    export PYTHONPATH=$PYTHONPATH:${MEGATRON_REPO_DIR}:${LLAMA_REPO_DIR}:${BIG_BENCH_REPO_DIR}:${MMLU_REPO_DIR} && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${MMLU_REPO_DIR}/evaluate.py ${TASK_OPTIONS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.
