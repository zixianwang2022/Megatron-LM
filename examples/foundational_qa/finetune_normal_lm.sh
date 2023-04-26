#!/bin/bash
# bash examples/qa/finetune_normal_lm.sh landrover_tasb_retrieved 843m 1 3e-6 1 


blend_name=$1
model_size=$2
global_bsz=$3
lr=$4
ft_neighbours=$5
model_card=$6
TASK=none

train_iters=100

. ./examples/foundational_qa/common_args.sh

num_nodes=1
num_gpus=8

min_lr=0.000001
if [[ $model_size == "8b" ]]; then
    num_nodes=4
    min_lr=0.0000001
fi

if [[ $model_size == "43b" ]]; then
    num_nodes=16
    min_lr=0.0000001
fi

SAVENAME="${blend_name}_${model_card}_same_format_ctx${ft_neighbours}_${model_size}_${global_bsz}_${lr}"
CHECKPOINT_PATH="${QA_HOME}/checkpoints/applications/${SAVENAME}"
TENSORBOARD_DIR="${QA_HOME}/tensorboard/${SAVENAME}"
mkdir -p ${TENSORBOARD_DIR}

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 50 \
             --eval-interval 2 \
             --tensorboard-dir ${TENSORBOARD_DIR} \
             --log-validation-ppl-to-tensorboard \
             --eval-iters 1"

DATA_BLEND="0.5 OBQA 0.5 PIQA"

options=" \
    $GPT_ARGS \
    --data-path ${DATA_BLEND} \
    --data-folder ${data_folder} \
    --sequence-parallel \
    --recompute-activations \
    --lr $lr \
    --micro-batch-size 1 \
    --global-batch-size ${global_bsz} \
    --min-lr ${min_lr} \
    --train-iters ${train_iters} \
    --dataloader-type cyclic \
    --save $CHECKPOINT_PATH \
    $OUTPUT_ARGS \
    $FT_ARGS"

if [[ -d "$CHECKPOINT_PATH" ]]; then
    options="$options \
        --load $CHECKPOINT_PATH "
else
    options="$options \
        --load $PRETRAINED_CHECKPOINT \
        --finetune \
        --no-load-optim "
fi

DIR=`pwd`
# -m torch.distributed.launch --nproc_per_node 8
run_cmd="python -m torch.distributed.launch --nproc_per_node 8 ${DIR}/tasks/foundational_QA/finetune_gpt_with_pretrain.py ${options}"
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


# submit_job --gpu ${num_gpus} --nodes ${num_nodes} --email_mode never  --mounts $MOUNTS --partition $PARTITION  --image $DOCKER -c "$LAUNCH ${run_cmd}" -n "${SAVENAME}" --duration 2 # --dependent_clones 4
${run_cmd}
