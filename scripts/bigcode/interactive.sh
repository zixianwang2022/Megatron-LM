#!/bin/bash

set -u

if [ "$#" != "3" ]; then
    echo "expected 3 args, found $#."
    exit 1
fi

######## Arguments. ########

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# . args.sh
# . /home/lmcafee/src/megatrons/megatron-lm-main/scripts/args.sh
. $DIR/../args_gen.sh "$@"

######## Task args. ########

# ALDR_DIR="/lustre/fsw/adlr/adlr-nlp"
# CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/mpatwary/checkpoints/gpt3/share-checkpoints/${MODEL_NAME}"
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
#     --tensor-model-parallel-size 4 \
#     --pipeline-model-parallel-size 1 \
#     --num-layers 32 \
#     --hidden-size 4096 \
#     --num-attention-heads 32 \
#     --seq-length 4096 \
#     --max-position-embeddings 4096 \
#     --micro-batch-size 1 \
#     --global-batch-size 256 \
#     --tokenizer-type GPTSentencePieceTokenizer \
#     --tokenizer-model /lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
#     --load ${CHECKPOINT_DIR} \
#     --bf16 \
#     --DDP-impl local "
# options=" \
#     ${imp_flags} \
#     --load ${CHECKPOINT_DIR} "

# MODEL="megatron"
# MODEL_NAME="gpt3-8b-multi-1.1t-gtc" 
MAX_GEN_LEN=256
BATCH_SIZE=5
TOP_K=0
TOP_P=0.9
TEMP=1.0
N_SAMPLES=200
N_CHUNKS=10 
# CHUNK_TO_EVAL="1 2 3 4 5 6 7 8 9 10" 
# TASKS="humaneval mbpp"

CHUNK_TO_EVAL="1"
TASK="humaneval"
# TASK="mbpp"

TAG="task-${TASK}_chunk-${CHUNK_TO_EVAL}-${N_CHUNKS}__${MODEL_FAMILY}-${MODEL_TYPE}-${MODEL_SIZE}"
# TAG="${MODEL_TYPE}-${MODEL_SIZE}-${TASK}_max_gen_len-${MAX_GEN_LEN}_batch_size-${BATCH_SIZE}_top_k-${TOP_K}_top_p-${TOP_P}_temp-${TEMP}_n_samples-${N_SAMPLES}_num_chunks-${NUM_CHUNKS}_chunk-${CHUNK_TO_EVAL}"
TASK_OPTIONS=" \
   --model=${MODEL_FAMILY} \
   --tasks=${TASK} \
   --max_length_generation=${MAX_GEN_LEN} \
   --bigcode_batch_size=${BATCH_SIZE} \
   --allow_code_execution=True \
   --top_k=${TOP_K} \
   --top_p=${TOP_P} \
   --temperature=${TEMP} \
   --n_samples=${N_SAMPLES} \
   --gen_token_save_iter=50 \
   --chunk_to_eval=${CHUNK_TO_EVAL} \
   --num_chunks=${N_CHUNKS} \
   --generations_save_path=generations__${TAG}.json \
   --output_path=results__${TAG}.json \
"
# ${MODEL}-${MODEL_NAME}-${TASK}_max_length_gen-${MAX_LENGTH_GEN}_bc_batch_size-${BATCH_SIZE}_top_k-${TOP_K}_top_p-${TOP_P}_temp-${TEMP}_n_samples-${N_SAMPLES}_chunk-${CHUNK_TO_EVAL}_num_chunks-${NUM_CHUNKS}
#    --generations_save_path=/workspace/bigcode-evaluation-harness/results/generations-${MODEL}-${MODEL_NAME}-${TASK}_max_length_gen-${MAX_LENGTH_GEN}_bc_batch_size-${BATCH_SIZE}_top_k-${TOP_K}_top_p-${TOP_P}_temp-${TEMP}_n_samples-${N_SAMPLES}_chunk-${CHUNK_TO_EVAL}_num_chunks-${NUM_CHUNKS}.json \
#    --output_path=/workspace/bigcode-evaluation-harness/results/evaluation-results-${MODEL}-${MODEL_NAME}-${TASK}_max_length_gen-${MAX_LENGTH_GEN}_bc_batch_size-${BATCH_SIZE}_top_k-${TOP_K}_top_p-${TOP_P}_temp-${TEMP}_n_samples-${N_SAMPLES}_chunk-${CHUNK_TO_EVAL}_num_chunks-${NUM_CHUNKS}.json "

# image=gitlab-master.nvidia.com/jupinderp/bigcode_containers:v2

# mount=${ALDR_DIR}:${ALDR_DIR},{YOUR_BIGCODE_EVAL_HARNESS_REPO_DIR}:/workspace/bigcode-evaluation-harness,{YOUR_MEGATRON_DIR}:/workspace/megatron-lm

######## Command. ########

# srun --container-image $image --container-mounts $mount bash -c "
#   export PYTHONPATH=/workspace/bigcode-evaluation-harness:/workspace/megatron-lm:\$PYTHONPATH;
#   cd /workspace/bigcode-evaluation-harness;
#   export CUDA_DEVICE_MAX_CONNECTIONS=1;
#   export NCCL_IB_SL=1;
#   set -x;
#   python /workspace/bigcode-evaluation-harness/main.py ${options} ${TASK_OPTIONS}
# "
CMD="\
    export PYTHONPATH=$PYTHONPATH:${MEGATRON_REPO_DIR}:${LLAMA_REPO_DIR}:${BIG_CODE_REPO_DIR} && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${BIG_CODE_REPO_DIR}/main.py ${ARGS} ${TASK_OPTIONS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof
