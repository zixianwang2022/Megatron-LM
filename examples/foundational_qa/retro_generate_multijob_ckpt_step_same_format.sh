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
ckpt=${12}
K=${13}

. ./examples/qa/common_args.sh

top_k=1
micro_bsz=1
SAMPLE_ARGS="--top_k $top_k"

if [[ $sampling == "beam" ]]; then
    micro_bsz=1
    SAMPLE_ARGS="--beam-search"
fi

#SAVENAME="${TASK}_${model_card}_same_format_ctx${ft_neighbours}_${model_size}_${global_bsz}_${lr}"
#CHECKPOINT_PATH="${QA_HOME}/checkpoints/applications/${SAVENAME}"
CHECKPOINT_PATH=${ckpt}
sample_output_file="${CHECKPOINT_PATH}/foundational_qa_${TASK}_${ft_neighbours}_${K}_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}.txt"

DIR=`pwd`

echo $sample_input_file
echo $sample_output_file

RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/boxinw/next-llm

GEN_ARGS="$SAMPLE_ARGS \
          --gen-start-idx $gen_start \
          --num-gen $num_gen \
          --ckpt-step ${ckpt_step} \
          --sample-input-file $sample_input_file \
          --sample-output-file $sample_output_file \
          --retro-workdir ${RETRO_WORKDIR} \
          --retro-add-retriever \
          --retro-num-neighbors ${K} \
          --use-retrieved-neighbours \
          "

DISTRIBUTED_ARGS="--nproc_per_node ${mod_par} \
                  --nnodes ${pip_par} \
                  --node_rank 0 \
                  --master_port 8889"

# COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/prompt_learning/text_generation.py \
# COMMAND="python -u ${DIR}/tasks/retro_qa/text_generation.py \

COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/tasks/foundational_QA/retro_text_generation.py"

if [[ $model_size == "43b" ]]; then
   COMMAND="$LAUNCH python -u ${DIR}/tasks/foundational_QA/retro_text_generation.py"
fi

COMMAND="$COMMAND \
       $GPT_ARGS \
       $GEN_ARGS \
       --load $CHECKPOINT_PATH \
       --micro-batch-size $micro_bsz \
       $FT_ARGS"

export SUBMIT_LOGS="${QA_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

QA_HOME="/lustre/fsw/adlr/adlr-nlp/boxinw/nvllm-qa"
MOUNTS="/lustre/fsw/adlr/adlr-nlp/"
PARTITION="luna"
LAUNCH="${ADLR_UTILS}/mp_launch"

submit_job --gpu ${mod_par} --nodes ${pip_par} --email_mode never  --mounts $MOUNTS --partition $PARTITION --image "/lustre/fsw/adlr/adlr-nlp/boxinw/images/retrov2.sqsh"  -c "$COMMAND" -n "generate_${model_size}_${TASK}" --duration 4
# $COMMAND
# -m torch.distributed.launch $DISTRIBUTED_ARGS 
