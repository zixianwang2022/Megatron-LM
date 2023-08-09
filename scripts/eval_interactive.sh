#!/bin/bash

set -u

if [ "$#" != "2" ]; then
    echo "expected 2 args, found $#."
    exit 1
fi

######## Arguments. ########

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# . args.sh
# . /home/lmcafee/src/megatrons/megatron-lm-main/scripts/args.sh
. $DIR/args_gen.sh "$@"

######## Task args. ########

# ADLR_DIR="/lustre/fsw/adlr/adlr-nlp"
# CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/mpatwary/checkpoints/gpt3/share-checkpoints/${MODEL_NAME}/tp8pp1"
# BPE_DIR="/lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/bpe"

# imp_flags=" \
#     --apply-layernorm-1p \
#     --untie-embeddings-and-output-weights \
#     --disable-bias-linear \
#     --no-position-embedding \
#     --use-rotary-position-embeddings \
#     --rotary-percent 0.5 \
#     --swiglu \
#     --attention-dropout 0.0 \
#     --hidden-dropout 0.0 \
#     --tensor-model-parallel-size 8 \
#     --pipeline-model-parallel-size 1 \
#     --num-layers 48 \
#     --hidden-size 8192 \
#     --num-attention-heads 64 \
#     --seq-length 4096 \
#     --max-position-embeddings 4096 \
#     --micro-batch-size 1 \
#     --global-batch-size 768 \
#     --tokenizer-type GPTSentencePieceTokenizer \
#     --tokenizer-model /lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
#     --load ${CHECKPOINT_DIR} \
#     --bf16 \
#     --DDP-impl local "

standard_tasks="abstract_narrative_understanding general_knowledge human_organs_senses intent_recognition riddle_sense similarities_abstraction simple_arithmetic_json simple_arithmetic_json_multiple_choice undo_permutation unit_conversion qa_wikidata linguistic_mappings date_understanding conlang_translation"
tydiqa_task_list="tydiqa_goldp.ar tydiqa_goldp.bn tydiqa_goldp.en tydiqa_goldp.fi tydiqa_goldp.id tydiqa_goldp.ko tydiqa_goldp.ru tydiqa_goldp.sw tydiqa_goldp.te"
task_lists="standard_tasks tydiqa_task_list"
declare -A my_dict
my_dict["standard_tasks"]=" --max_length=64 --json_shots='0,1,2' "
my_dict["tydiqa_task_list"]=" --max_length=16 --json_shots='1,4' "

TASK_LIST="standard_tasks"
TASK="abstract_narrative_understanding"
# TASK_ARGS="--task=${TASK} ${my_dict[$TASK_LIST]}",MODEL_NAME="megatron-llama"
ARGS=" \
    ${ARGS} \
    --top_k 1 \
    --top_p 0.0 "
# ${TASK_ARGS} \
TASK_OPTIONS=" \
   --models=megatron-lm \
   --task=${TASK} \
   ${my_dict[$TASK_LIST]} \
   --undefok=sequence-parallel,recompute-activations,use-flash-attn,overlap-p2p-communication,apply-layernorm-1p,untie-embeddings-and-output-weights,disable-bias-linear,no-position-embedding,use-rotary-position-embeddings,rotary-percent,swiglu,attention-dropout,hidden-dropout,exit-duration-in-mins,tensor-model-parallel-size,pipeline-model-parallel-size,num-layers,hidden-size,num-attention-heads,seq-length,max-position-embeddings,micro-batch-size,global-batch-size,rampup-batch-size,train-samples,lr-decay-samples,lr-warmup-samples,lr,min-lr,lr-decay-style,log-interval,eval-iters,eval-interval,tokenizer-type,tokenizer-model,save-interval,load,split,clip-grad,weight-decay,adam-beta1,adam-beta2,init-method-std,log-params-norm,log-num-zeros-in-grad,bf16,DDP-impl,top_k,top_p,num-layers-per-virtual-pipeline-stage\
\
,norm-epsilon,no-masked-softmax-fusion,load-llama,no-load-optim,no-load-rng,fp16,gen-model,norm-type,exit-on-missing-checkpoint,use-checkpoint-args,no-query-key-layer-scaling"
# please note that undefok needs to be defined properly by including all flags added in $options.
# this solve the conflicts between `argparse` and `absl.flags`

# image=gitlab-master.nvidia.com/jupinderp/bigbench_containers:v1

# mount=${ADLR_DIR}:${ADLR_DIR},${BPE_DIR}:${BPE_DIR},${BIG_BENCH_DIR}:/workspace/big-bench-megatron-lm,${MEGATRON_DIR}:/workspace/megatron-lm

# srun --container-image $image --container-mounts $mount bash -c "
#   export PYTHONPATH=/workspace/big-bench-megatron-lm:/workspace/megatron-lm:\$PYTHONPATH;
#   cd /workspace/big-bench-megatron-lm;
#   export CUDA_DEVICE_MAX_CONNECTIONS=1;
#   export NCCL_IB_SL=1;
#   pip install einops;
#   pip install sacrebleu --upgrade;
#   echo ${TASK};
#   set -x;
#   python /workspace/big-bench-megatron-lm/bigbench/evaluate_task.py ${options} ${TASK_OPTIONS}
# "
######## Command. ########

# NPROCS=1 # 8
# CMD="\
#     cd ${MEGATRON_REPO_DIR} && \
#     export PYTHONPATH=$PYTHONPATH:${MEGATRON_REPO_DIR}:${LLAMA_REPO_DIR}:/home/lmcafee/src && \
#     python -m torch.distributed.run \
#     --nproc_per_node ${NPROCS} \
#     --nnodes 1 \
#     --node_rank ${NODE_RANK} \
#     --master_addr ${MASTER_ADDR} \
#     --master_port 6000 \
#     ${SCRIPT} ${ARGS} \
# "
CMD="\
    cd ${MEGATRON_REPO_DIR} && \
    export PYTHONPATH=$PYTHONPATH:${MEGATRON_REPO_DIR}:${LLAMA_REPO_DIR}:${BIG_BENCH_REPO_DIR} && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${BIG_BENCH_REPO_DIR}/bigbench/evaluate_task.py ${ARGS} ${TASK_OPTIONS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.
