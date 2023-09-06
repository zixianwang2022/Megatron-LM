#!/bin/bash

TASK=$1
model_size=$2
sampling=$3
split=$4
gen_start=$5
num_gen=$6
ckpt_step=$7
ft_neighbours=$8
SAVENAME=$9
model_card=$9
use_retrieved_neighbours=${10}

. ./examples/long_context_flan/long_context_args.sh
. ./examples/long_context_flan/gen_input.sh

top_k=1
micro_bsz=1
SAMPLE_ARGS="--top_k $top_k"

if [[ $sampling == "beam" ]]; then
    micro_bsz=1
    SAMPLE_ARGS="--beam-search"
fi

CHECKPOINT_PATH="${QA_HOME}/checkpoints/applications/${SAVENAME}"
sample_output_file="${CHECKPOINT_PATH}/${TASK}_${ft_neighbours}_generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}.txt"

if [[ $use_retrieved_neighbours ]]; then
    sample_output_file="${CHECKPOINT_PATH}/${TASK}_${ft_neighbours}_generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}_ret.txt"
fi

DIR=`pwd`

GEN_ARGS="$SAMPLE_ARGS \
          --gen-start-idx $gen_start \
          --num-gen $num_gen \
          --ckpt-step ${ckpt_step} \
	  --ft_neighbours ${ft_neighbours} \
          --sample-input-file $sample_input_file \
          --sample-output-file $sample_output_file"

if [[ ${model_card} == *itp-32k*  ]]; then 
	mod_par=8
	pip_par=8
fi
DISTRIBUTED_ARGS="--nproc_per_node ${mod_par} \
                  --nnodes ${pip_par} \
                  --node_rank 0 \
                  --master_port 8889"

# COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/prompt_learning/text_generation.py \
# COMMAND="python -u ${DIR}/tasks/retro_qa/text_generation.py \

COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/tasks/long_context_QA/text_generation.py"

if [[ $model_size == "43b" ]]; then
   COMMAND="$LAUNCH python -u ${DIR}/tasks/long_context_QA/text_generation.py"
fi

COMMAND="$COMMAND \
       $GPT_ARGS \
       $GEN_ARGS \
       --load $CHECKPOINT_PATH \
       --micro-batch-size $micro_bsz"

if [[ $use_retrieved_neighbours ]]; then
        COMMAND+=" --use-retrieved-neighbours "
fi

export SUBMIT_LOGS="${QA_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo $COMMAND
# export SUBMIT_ACCOUNT=llmservice_nlp_fm
submit_job --gpu ${mod_par} --nodes ${pip_par} --email_mode never  --mounts $MOUNTS --partition $PARTITION --image $DOCKER  -c "$COMMAND" -n "generate_cross_${model_size}_${TASK}" --duration 2
# -m torch.distributed.launch $DISTRIBUTED_ARGS 
