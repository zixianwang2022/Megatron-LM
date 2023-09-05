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

. ./examples/foundational_qa/common_args.sh
. ./examples/foundational_qa/gen_input.sh

top_k=1
micro_bsz=1
SAMPLE_ARGS="--top_k $top_k"

if [[ $sampling == "beam" ]]; then
    micro_bsz=1
    SAMPLE_ARGS="--beam-search"
fi

# CHECKPOINT_PATH="/lustre/fsw/adlr/adlr-nlp/pengx/long_context_llm/megatron-lm/checkpoints/gpt3-43b-mult"
# CHECKPOINT_PATH="/lustre/fsw/adlr/adlr-nlp/pengx/long_context_llm_with_broken_main/megatron-lm/checkpoints/gpt3-43b-multi-1.1t-gtc-itp-16k-lr1e-5/"
# CHECKPOINT_PATH="/lustre/fsw/adlr/adlr-nlp/pengx/long_context_llm_with_broken_main/megatron-lm/checkpoints/gpt3-8b-multi-1.1t-gtc-itp-16k-lr1e-5/"
CHECKPOINT_PATH="/lustre/fsw/adlr/adlr-nlp/pengx/sft_43b_qa/megatron-lm/checkpoints/gpt3-8b-multi-1.1t-gtc-itp-16k-lr1e-5/"
CHECKPOINT_PATH="/lustre/fsw/adlr/adlr-nlp/pengx/long_context_llm_with_broken_main/megatron-lm/checkpoints/gpt3-8b-multi-1.1t-gtc-4xbsz"
# CHECKPOINT_PATH="/lustre/fsw/adlr/adlr-nlp/pengx/long_context_llm_with_broken_main/megatron-lm/checkpoints/gpt3-8b-multi-1.1t-gtc-base/"
sample_output_file="${CHECKPOINT_PATH}/${TASK}_${ft_neighbours}_generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}.txt.sft.${model_card}"

if [[ $use_retrieved_neighbours ]]; then
    sample_output_file="${CHECKPOINT_PATH}/${TASK}_${ft_neighbours}_generate_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}_ret.txt.sft.${model_card}"
fi

sample_output_file="${sample_output_file}.v2"

DIR=`pwd`

GPT_ARGS="--apply-layernorm-1p \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --rotary-percent 0.5 \
        --swiglu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --pipeline-model-parallel-size $pip_par \
        --tensor-model-parallel-size $mod_par \
        --num-layers $layers \
        --hidden-size $hid_dim \
        --num-attention-heads $heads \
        --seq-length 16384 \
        --max-position-embeddings 16384 \
        --rotary-seq-len-interpolation-factor 4 \
        --lr-decay-style cosine \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --bf16 \
        --DDP-impl local"

# GPT_ARGS="--apply-layernorm-1p \
#         --untie-embeddings-and-output-weights \
#         --disable-bias-linear \
#         --no-position-embedding \
#         --use-rotary-position-embeddings \
#         --rotary-percent 0.5 \
#         --swiglu \
#         --attention-dropout 0.0 \
#         --hidden-dropout 0.0 \
#         --pipeline-model-parallel-size $pip_par \
#         --tensor-model-parallel-size $mod_par \
#         --num-layers $layers \
#         --hidden-size $hid_dim \
#         --num-attention-heads $heads \
#         --seq-length 4096 \
#         --max-position-embeddings 4096 \
#         --lr-decay-style cosine \
#         --tokenizer-type GPTSentencePieceTokenizer \
#         --tokenizer-model ${TOKENIZER_MODEL} \
#         --clip-grad 1.0 \
#         --weight-decay 0.1 \
#         --adam-beta1 0.9 \
#         --adam-beta2 0.95 \
#         --log-params-norm \
#         --log-num-zeros-in-grad \
#         --bf16 \
#         --DDP-impl local"

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

COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/tasks/foundational_QA/text_generation_conv.py"

if [[ $model_size == "43b" ]]; then
   COMMAND="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${DIR}/tasks/foundational_QA/text_generation_conv.py"
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

$COMMAND
# -m torch.distributed.launch $DISTRIBUTED_ARGS 
