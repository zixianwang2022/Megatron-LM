#!/bin/bash
# bash examples/qa/finetune_normal_lm.sh landrover_tasb_retrieved 843m 1 3e-6 1 


blend_name=$1
model_size=$2
global_bsz=$3
lr=$4
ft_neighbours=$5
model_card=$6
TASK=none

train_iters=15000

. ./examples/fqa_llama2/common_args_llama2.sh

num_nodes=1
num_gpus=8

min_lr=0.000001

## for nvllm
if [[ $model_size == "8b" ]]; then
    num_nodes=4
    min_lr=0.00000001
fi

if [[ $model_size == "43b" ]]; then
    num_nodes=64
    # num_nodes=4 # debug
    min_lr=0.00000001
fi

## for llama
if [[ $model_size == "7b" ]]; then
    num_nodes=2 # debug
    # num_nodes=4
    min_lr=0.00000001
fi

if [[ $model_size == "13b" ]]; then
    num_nodes=8
    min_lr=0.00000001
fi

if [[ $model_size == "70b" ]]; then
    num_nodes=64
    # num_nodes=8 # debug
    min_lr=0.00000001
fi


SAVENAME="${blend_name}_${model_card}_same_format_fewshot_ctx${ft_neighbours}_${model_size}_${global_bsz}_${lr}"
CHECKPOINT_PATH="${QA_HOME}/checkpoints/applications/${SAVENAME}"
TENSORBOARD_DIR="${QA_HOME}/tensorboard/${SAVENAME}"
mkdir -p ${TENSORBOARD_DIR}

## original --save-interval 1500
## make --save-interval 1200 so that we can save two checkpoints in 4 hrs
OUTPUT_ARGS="--log-interval 10 \
             --save-interval 1200 \
             --eval-interval 100 \
             --tensorboard-dir ${TENSORBOARD_DIR} \
             --log-validation-ppl-to-tensorboard \
             --eval-iters 200"

. ./examples/foundational_qa/${blend_name}.sh
. ./examples/foundational_qa/fewshot_train_files.sh

options=" \
    $GPT_ARGS \
    --data-path ${DATA_BLEND} \
    --data-folder ${data_folder} \
    --fewshot-files ${FEWSHOT_FILES} \
    --max-num-shot 5 \
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

# --sequence-parallel \     # remove this for llama-2 finetuning

if [[ $model_card == *llama2_chat_7b* ]]; then
    PRETRAINED_CHECKPOINT=$llama2_chat_7b
fi
if [[ $model_card == *llama2_text_7b* ]]; then
    PRETRAINED_CHECKPOINT=$llama2_text_7b
fi
if [[ $model_card == *llama2_chat_13b* ]]; then
    PRETRAINED_CHECKPOINT=$llama2_chat_13b
fi
if [[ $model_card == *llama2_text_13b* ]]; then
    PRETRAINED_CHECKPOINT=$llama2_text_13b
fi
if [[ $model_card == *llama2_chat_70b* ]]; then
    PRETRAINED_CHECKPOINT=$llama2_chat_70b
fi
if [[ $model_card == *llama2_text_70b* ]]; then
    PRETRAINED_CHECKPOINT=$llama2_text_70b
fi
if [[ $model_card == *llama2_text_70b_with_qc* ]]; then
    PRETRAINED_CHECKPOINT=$llama2_text_70b_with_qc
fi


if [[ -f "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" ]]; then
    options="$options \
        --load $CHECKPOINT_PATH "
else
    options="$options \
        --load $PRETRAINED_CHECKPOINT \
        --finetune \
	    --no-load-rng \
        --no-load-optim "
fi

DIR=`pwd`
# -m torch.distributed.launch --nproc_per_node 8
run_cmd="python -u ${DIR}/tasks/foundational_QA/finetune_gpt_with_pretrain.py ${options}"
# srun -l \
#      --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/faissgpu" \
#      --container-mounts "/home/pengx/projects/retro/:/home/pengx/projects/retro/" \
#      --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"
# $run_cmd

## running command
# bash examples/fqa_llama2/finetune_llama2_fewshot.sh multiturn_qa_blendv2 70b 64 3e-7 1 llama2_text_70b_with_qc_multiturn


export SUBMIT_LOGS="${QA_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


echo ${run_cmd}
submit_job --gpu ${num_gpus} --nodes ${num_nodes} --email_mode never  --mounts $MOUNTS --partition $PARTITION  --image $DOCKER -c "$LAUNCH ${run_cmd}" -n "${SAVENAME}" --duration 4 --exclude luna-0534,luna-0253,luna-0377  # --dependent_clones 1
