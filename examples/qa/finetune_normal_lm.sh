#!/bin/bash
# bash examples/qa/finetune_normal_lm.sh landrover_tasb_retrieved 843m 1 3e-6 1 


TASK=$1
model_size=$2
bsz=$3
lr=$4
ft_neighbours=$5
model_card=$6

. ./examples/qa/common_args.sh

epochs=5
num_nodes=1
num_gpus=8

min_lr=0.000001
if [[ $model_size == "8b" ]]; then
    num_nodes=4
    min_lr=0.0000001
fi

if [[ $model_size == "43b" ]]; then
    num_nodes=4
    min_lr=0.0000001
fi

global_bsz=$((num_gpus*bsz*num_nodes/mod_par/pip_par))
SAVENAME="${TASK}_${model_card}_same_format_ctx${ft_neighbours}_${model_size}_${global_bsz}_${lr}"
CHECKPOINT_PATH="${QA_HOME}/checkpoints/applications/${SAVENAME}"
TENSORBOARD_DIR="${CHECKPOINT_PATH}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}

OUTPUT_ARGS="--log-interval 20 \
             --save-interval 1500 \
             --eval-interval 300 \
             --tensorboard-dir ${TENSORBOARD_DIR} \
             --log-validation-ppl-to-tensorboard \
             --eval-iters 100"

options=" \
    $GPT_ARGS \
    --sequence-parallel \
    --recompute-activations \
    --lr $lr \
    --micro-batch-size $bsz \
    --min-lr ${min_lr} \
    --epochs $epochs \
    --save $CHECKPOINT_PATH \
    $OUTPUT_ARGS \
    $FT_ARGS \
    --finetune \
	--no-load-rng \
    --no-load-optim \
    --pretrained-checkpoint $PRETRAINED_CHECKPOINT"

DIR=`pwd`
# -m torch.distributed.launch --nproc_per_node 8
run_cmd="python -u ${DIR}/tasks/retro_qa/main.py ${options}"
# srun -l \
#      --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/faissgpu" \
#      --container-mounts "/home/pengx/projects/retro/:/home/pengx/projects/retro/" \
#      --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"
# $run_cmd

export SUBMIT_LOGS="${QA_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "submit_job --gpu ${num_gpus} --nodes ${num_nodes} --email_mode never  --mounts $MOUNTS --partition $PARTITION  --image $DOCKER -c "$LAUNCH ${run_cmd}" -n "${SAVENAME}" --duration 2"
submit_job --gpu ${num_gpus} --nodes ${num_nodes} --email_mode never  --mounts $MOUNTS --partition $PARTITION  --image $DOCKER -c "$LAUNCH ${run_cmd}" -n "${SAVENAME}" --duration 2 # --dependent_clones 4
# ${run_cmd}
