#!/bin/bash
exit ## retire
TASK=$1
model_size=$2
sampling=$3
split=$4
gen_start=$5
num_gen=$6
ft_neighbours=$8
use_retrieved_neighbours=$9
model_card="unbiased_cuckoo_pp1"
model_card="quiet_cockatoo_pp1"

. ./examples/foundational_qa/common_args.sh
. ./examples/foundational_qa/gen_input.sh

top_k=1
micro_bsz=1
SAMPLE_ARGS="--top_k $top_k"

if [[ $sampling == "beam" ]]; then
    micro_bsz=1
    SAMPLE_ARGS="--beam-search"
fi

if [[ $model_card == *unbiased_cuckoo* ]]; then
    CHECKPOINT_PATH=${unbiased_cuckoo}
fi
if [[ $model_card == *quiet_cockatoo* ]]; then
    CHECKPOINT_PATH=${quiet_cockatoo}
fi

sample_output_file="${CHECKPOINT_PATH}/${TASK}_${ft_neighbours}_generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}.txt"
if [[ $use_retrieved_neighbours ]]; then
    sample_output_file="${CHECKPOINT_PATH}/${TASK}_${ft_neighbours}_generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_ret.txt"
fi

DIR=`pwd`

GEN_ARGS="$SAMPLE_ARGS \
          --gen-start-idx $gen_start \
          --num-gen $num_gen \
          --sample-input-file $sample_input_file \
          --sample-output-file $sample_output_file"

DISTRIBUTED_ARGS="--nproc_per_node ${mod_par} \
                  --nnodes ${pip_par} \
                  --node_rank 0 \
                  --master_port 8889"

# COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/prompt_learning/text_generation.py \
# COMMAND="python -u ${DIR}/tasks/retro_qa/text_generation.py \

COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/tasks/foundational_QA/text_generation.py"

if [[ $model_size == "43b" ]]; then
   COMMAND="$LAUNCH python -u ${DIR}/tasks/foundational_QA/text_generation.py"
fi

COMMAND="$COMMAND \
       $GPT_ARGS \
       $GEN_ARGS \
       --load $CHECKPOINT_PATH \
       --micro-batch-size $micro_bsz \
       $FT_ARGS"

if [[ $use_retrieved_neighbours ]]; then
	COMMAND+=" --use-retrieved-neighbours "
fi

export SUBMIT_LOGS="${QA_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

submit_job --gpu ${mod_par} --nodes ${pip_par} --email_mode never  --mounts $MOUNTS --partition $PARTITION --image $DOCKER  -c "$COMMAND" -n "generate_zeroshot_${model_size}_${TASK}" --duration 1
echo $COMMAND
# -m torch.distributed.launch $DISTRIBUTED_ARGS 
