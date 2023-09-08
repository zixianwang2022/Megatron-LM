#!/bin/bash

TASK=$1
model_size=$2
sampling=$3
split=$4
global_bsz=$5
lr=$6
dropout=0.0
gen_start=$7
num_gen=$8
ckpt_step=${9}
ft_neighbours=${10}
model_card=${11}

. ./examples/qa/common_args.sh

top_k=1
micro_bsz=1
SAMPLE_ARGS="--top_k $top_k"

if [[ $sampling == "beam" ]]; then
    micro_bsz=1
    SAMPLE_ARGS="--beam-search"
fi

CHECKPOINT=$PRETRAINED_CHECKPOINT
SAVENAME="${TASK}_${model_card}_same_format_ctx${ft_neighbours}_${model_size}_${global_bsz}_${lr}"
CHECKPOINT_PATH="${QA_HOME}/checkpoints/applications/${SAVENAME}"
SAMPLE_OUTPUT_PATH=".."

#Use task checkpoint instead of pretrained checkpoint if path exists
if [ -e $CHECKPOINT_PATH ]; then
    CHECKPOINT=$CHECKPOINT_PATH
    SAMPLE_OUTPUT_PATH=$CHECKPOINT_PATH
    sample_output_file="${SAMPLE_OUTPUT_PATH}/generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}.txt"
else
    echo "Execution of non-finetuned checkpoint text generation using ${CHECKPOINT}"
    sample_output_file="${SAMPLE_OUTPUT_PATH}/generate_${TASK}_${model_card}.txt"
fi

#Before submitting the job make sure the checkpoint step exists in the checkpoint path
if ! find "${CHECKPOINT}" -type d -name "iter*${ckpt_step}" -print | grep -q "."; then
    echo "The checkpoint step provided (${ckpt_step}) does not exist at the checkpoint path: ${CHECKPOINT}"
    exit 1
fi

#sample_output_file="${SAMPLE_OUTPUT_PATH}/generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}.txt"
#sample_output_file="${CHECKPOINT_PATH}/generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}.txt"

DIR=`pwd`

GEN_ARGS="$SAMPLE_ARGS \
          --gen-start-idx $gen_start \
          --num-gen $num_gen \
          --ckpt-step ${ckpt_step} \
          --sample-input-file $sample_input_file \
          --sample-output-file $sample_output_file"

DISTRIBUTED_ARGS="--nproc_per_node ${mod_par} \
                  --nnodes ${pip_par} \
                  --node_rank 0 \
                  --master_port 8889"

# COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/prompt_learning/text_generation.py \
# COMMAND="python -u ${DIR}/tasks/retro_qa/text_generation.py \

COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/tasks/retro_qa/text_generation.py"

if [[ $model_size == "43b" ]]; then
    COMMAND="$LAUNCH python -u ${DIR}/tasks/retro_qa/text_generation.py"
    for CONV_PROMPT_LIST in *-pp1-v3 *-pp1-v4 *-pp1-v5 *-pp1-v6 *-pp1-v7 *-pp1-v8
    do
        if [[ ${model_card} == $CONV_PROMPT_LIST ]]; then
            COMMAND="$LAUNCH python -u ${DIR}/tasks/retro_qa/text_generation_conv.py"
        fi
    done
fi

COMMAND="$COMMAND \
       $GPT_ARGS \
       $GEN_ARGS \
       --load $CHECKPOINT \
       --micro-batch-size $micro_bsz \
       $FT_ARGS"

echo $COMMAND

export SUBMIT_LOGS="${QA_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

submit_job --gpu ${mod_par} --nodes ${pip_par} --email_mode never  --mounts $MOUNTS --partition $PARTITION --image $DOCKER  -c "$COMMAND" -n "generate_${model_size}_${TASK}" --duration 1
# $COMMAND
# -m torch.distributed.launch $DISTRIBUTED_ARGS 
