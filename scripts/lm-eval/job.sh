#!/bin/bash

#SBATCH -p batch_block1,batch_block2
#SBATCH --nodes=1
#SBATCH -A adlr
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:llama
#SBATCH --ntasks-per-node=1

# ... SBATCH --dependency=singleton

set -u

######## Arguments. ########

. ${SCRIPT_DIR}/../args_gen.sh ${MODEL_FAMILY} ${MODEL_TYPE} ${MODEL_SIZE}

######## Task args. ########

# --no_cache \
TASK_OPTIONS=" \
   --tasks=${TASK} \
   --model ${MODEL_FAMILY} \
   --system_prompt_prefix '' \
   --input_prompt_prefix '' \
   --output_prompt_prefix '' \
   ${ARGS} \
"

######## Command. ########

IMAGE=nvcr.io/nvidia/pytorch:23.04-py3
MOUNT=/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2:/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2,/home/lmcafee/src/lutil:/home/lmcafee/src/lutil

srun \
    --container-image ${IMAGE} \
    --container-mounts ${MOUNT} \
    bash -c "
  export PYTHONPATH=${MEGATRON_REPO_DIR}:${LLAMA_REPO_DIR}:${LM_EVAL_REPO_DIR}:/home/lmcafee/src;
  export CUDA_DEVICE_MAX_CONNECTIONS=1;
  export NCCL_IB_SL=1;
  pip install fairscale;
  pip install sentencepiece;
  pip install -U transformers;
  pip install accelerate;
  pip install lm-eval;
  set -x;
  python ${LM_EVAL_REPO_DIR}/main.py ${TASK_OPTIONS}
"

# eof.
