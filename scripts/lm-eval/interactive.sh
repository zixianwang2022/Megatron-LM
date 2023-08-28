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

######## Task args. ########

# TASKS="boolq,piqa,hellaswag,winogrande"
TASKS="boolq"
# TASKS="piqa"
# TASKS="hellaswag"
# TASKS="winogrande"

# ARGS=" \
#     ${ARGS} \
#     --top_k 1 \
#     --top_p 0.0 "
TASK_OPTIONS=" \
   --tasks=${TASKS} \
   --model ${MODEL_FAMILY} \
   --no_cache \
   --system_prompt_prefix '' \
   --input_prompt_prefix '' \
   --output_prompt_prefix '' \
   ${ARGS} \
"
#   --model_args '${ARGS}' \

# please note that undefok needs to be defined properly by including all flags added in $options.
# this solve the conflicts between `argparse` and `absl.flags`

# image=gitlab-master.nvidia.com/jupinderp/bigbench_containers:v1

# mount=${ADLR_DIR}:${ADLR_DIR},${BPE_DIR}:${BPE_DIR},${BIG_BENCH_DIR}:/workspace/big-bench-megatron-lm,${MEGATRON_DIR}:/workspace/megatron-lm

# srun --container-image $image --container-mounts $mount bash -c "
#   export PYTHONPATH=/workspace/big-bench-megatron-lm:/workspace/megatron-lm:\$PYTHONPATH;
#   cd /workspace/big-bench-megatron-lm;
#   export CUDA_DEVICE_MAX_CONNECTIONS=1;
#   export NCCL_IB_SL=1;
#   pip install einops;
#   pip install sacrebleu --upgrade;
#   echo ${TASK};
#   set -x;
#   python /workspace/big-bench-megatron-lm/bigbench/evaluate_task.py ${options} ${TASK_OPTIONS}
# "
######## Command. ########

# cd ${MEGATRON_REPO_DIR} && \
CMD="\
    export PYTHONPATH=$PYTHONPATH:${MEGATRON_REPO_DIR}:${LLAMA_REPO_DIR}:${BIG_BENCH_REPO_DIR} && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${LM_EVAL_REPO_DIR}/main.py ${TASK_OPTIONS} \
"
# >>>
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD
# +++
# <<<

# eof.
