#!/bin/bash

#SBATCH -p batch_dgx2h_m2 --nodes=8 --gres=gpu:16 -A gpu_adlr_nlp -t 1:00:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retro --ntasks-per-node=16 --dependency=singleton

set -u

DIR=$(readlink -f `pwd`)
LOG_DIR=$DIR/tools/retro/index/sandbox/logs
mkdir -p $LOG_DIR

REPO="retro-preprocess-play"
MODEL_KEY="megatron"
# MODEL_KEY="huggingface"

run_cmd=" \
    pwd && cd $SHARE_SOURCE/megatrons/megatron-lm-${REPO} && pwd && \
    export PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-${REPO}&&\
    python -um tools.retro.index.sandbox.megatron_vs_huggingface ${MODEL_KEY}
"

# >>>
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo $run_cmd
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# <<<


# IMAGE="nvcr.io#nvidia/pytorch:21.12-py3"
IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/faissgpu"
CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/gpfs/fs1/projects/gpu_adlr/datasets:/gpfs/fs1/projects/gpu_adlr/datasets"
srun -l \
     --container-image $IMAGE \
     --container-mounts $CONTAINER_MOUNTS \
     --output=$LOG_DIR/"%j_$RETRO_TASKS.log" \
     sh -c "${run_cmd}"
