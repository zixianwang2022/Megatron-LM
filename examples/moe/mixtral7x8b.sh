#!/bin/bash
#SBATCH -p batch
#SBATCH -A coreai_dlalgo_llm
#SBATCH -J coreai_dlalgo_llm-FW:mixtral_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -t 0:15:00
#SBATCH -o my_output.log

# Change the following paths via ENV variables to point to your data and Megatron code directories
MEGATRON_PATH=/lustre/fsw/coreai_dlalgo_llm/yihuih/moe-mlm
#/lustre/fsw/coreai_devtech_all/denliu/MoE-dev/megatron-lm-moe
OUTPUT_PATH=${MEGATRON_PATH}/output
#/lustre/fsw/coreai_devtech_all/denliu/MoE-dev/megatron-moe-scripts
ADLR_SHARING=/lustre/share/llmservice_nlp_fm/adlr-nlp-sharing

echo "MEGATRON_PATH: $MEGATRON_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "ADLR_SHARING: $ADLR_SHARING"

# Load necessary modules and set environment variables
# . $ADLR_SHARING/nvllm-1.1t/data/tokens/multi-1.1t-gtc-blend-v0.1.sh
DATA_BLEND=/lustre/fsw/coreai_dlalgo_llm/denliu/datasets/datasets/wudao_mistralbpe_content_document
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

# Set up directories and file paths
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
WORLD_SIZE=$(($SLURM_JOB_NUM_NODES*$SLURM_NTASKS_PER_NODE))

CHECKPOINT_PATH=$OUTPUT_PATH/checkpoints/
TENSORBOARD_PATH=$OUTPUT_PATH/tensorboard/
NSYS_PATH=$OUTPUT_PATH/nsys/
LOGS_PATH=$OUTPUT_PATH/logs-benchmark-pretrained/
mkdir -p ${CHECKPOINT_PATH} ${TENSORBOARD_PATH} ${NSYS_PATH} ${LOGS_PATH} ${OUTPUT_PATH}/cache

# Set model and training parameters
MOCK=${MOCK:-false} # if use mocked data and mixtral original tokenizer, else use nvllm tokenizer

PROFILE=0 # if enable nsys profile
MODEL_SIZE=7B
MODEL_TYPE=MIXTRAL # GPT, LLaMa-2, NVLLM
TP=${TP:-8}
PP=${PP:-1}
MBS=${MBS:-1}
GBS=${GBS:-256}
SL=4096 # sequence length. mixtral use 32768
WANDB_API_KEY=b1d8825af2c256485e86683005098aaea7a6157b # Use your own key
PR=${PR:-bf16} # precision

# Set MOE parameters
NUM_EXPERTS=8
EP=${EP:-1}
MOE_BALANCING_TYPE=aux_loss
MOE_TOPK=2

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
FFN_HIDDEN_SIZE=14336
LR=1.2e-4
MIN_LR=1.2e-5

GPT_MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_ATTN_HEADS 
    --seq-length $SL
    --max-position-embeddings 32768
    --ffn-hidden-size $FFN_HIDDEN_SIZE
)

MODEL_SPECIFIC_ARGS=(
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

TRAINING_ARGS=(
    --use-flash-attn 
    --untie-embeddings-and-output-weights 
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-samples 268554688
    --lr-decay-samples 255126953
    --lr-warmup-samples 162761
    --lr $LR 
    --min-lr $MIN_LR
    --lr-decay-style cosine 
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.008
    --bf16
    --use-mcore-models
)

if [ "$MOCK" == "true" ]; then
    TRAINING_ARGS+=(
        --tokenizer-type Llama2Tokenizer
        --tokenizer-model /lustre/fsw/coreai_dlalgo_llm/denliu/checkpoints/mixtral-hf/tokenizer.model
        --mock-data
    )
else
    TRAINING_ARGS+=(
        # --tokenizer-type GPTSentencePieceTokenizer
        # --tokenizer-model /lustre/fsw/coreai_devtech_all/denliu/MoE-dev/datasets/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
        --tokenizer-type Llama2Tokenizer
        --tokenizer-model /lustre/fsw/coreai_devtech_all/denliu/MoE-dev/checkpoints/mixtral-hf/tokenizer.model
    )
fi

if [ "$PR" == "fp8" ]; then
    TRAINING_ARGS+=(
        --fp8-format hybrid
        --fp8-amax-history-len 1024
        --fp8-amax-compute-algo max
    )
fi

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP 
    --sequence-parallel
    --use-distributed-optimizer
)

DATA_ARGS=(
    --data-path ${DATA_BLEND} 
    --data-cache-path ${OUTPUT_PATH}/cache
    --split 99,1,0
)

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}   
    --expert-model-parallel-size ${EP}
    --moe-router-load-balancing-type ${MOE_BALANCING_TYPE}
    --moe-router-topk ${MOE_TOPK}
    --moe-aux-loss-coeff 1e-2
)

if [ $PR == "bf16" ]; then
    MOE_ARGS+=(
        --moe-grouped-gemm
    )
fi


NAME="${MODEL_TYPE}-${MODEL_SIZE}-${PR}_GPUS${WORLD_SIZE}_TP${TP}_PP${PP}_EP${EP}_MBS${MBS}_GBS${GBS}_SEQ${SL}_EXPERT${NUM_EXPERTS}_${MOE_BALANCING_TYPE}_top${MOE_TOPK}_LR${LR}_EOS_$1"

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1000
    --eval-interval 100
    --eval-iters 32
    --save ${CHECKPOINT_PATH}/${NAME}
    # --load ${CHECKPOINT_PATH}/${NAME}
    # --load /lustre/fsw/coreai_dlalgo_llm/denliu/checkpoints/mixtral-mcore-TP${TP}PP${PP}EP${EP}
    ## Noted by Shiqing: add back for finetune
    # --finetune
    
    --tensorboard-dir ${TENSORBOARD_PATH}/${NAME}
    --tensorboard-queue-size 100
    --log-timers-to-tensorboard
    --log-batch-size-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-num-zeros-in-grad 
    --log-throughput
    --distributed-timeout-minutes 6000 
    --exit-duration-in-mins 230 \
)

WANDB_NAME=Mixtral-8x7B-roadmap
#WANDB_NAME=Mixtral-Finetune-Divergence-Reproduce

if [ -n "${WANDB_API_KEY}" ]; then
    export WANDB_API_KEY=${WANDB_API_KEY}
    EVAL_AND_LOGGING_ARGS+=(
        --wandb-project $WANDB_NAME
        --wandb-exp-name ${NAME}
    )
fi

# Profiling command

if [ "${PROFILE}" = 1 ]; then
PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --stats true --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage true -f true -x true -o ${NSYS_DIR}/${NAME}-${TIME}"
 
MEGATRON_EXTRA_PARAMS="${MEGATRON_EXTRA_PARAMS} --profile --profile-step-start 10 --profile-step-end 11 --profile-ranks 0 "
else
PROFILE_CMD=""
MEGATRON_EXTRA_PARAM=""
fi
echo "PROFILE_CMD="
echo $PROFILE_CMD
echo "MEGATRON_EXTRA_PARAM="
echo $MEGATRON_EXTRA_PARAM

# Run command
# ([[ "\$SLURM_LOCALID" == "0" ]] && echo "installing" && pip install --no-cache-dir wandb sentencepiece git+https://github.com/fanshiqing/grouped_gemm@main) ; ([[ "\$SLURM_LOCALID" != "0" ]] && echo "sleeping" && sleep 240) ;

DIR=/home/yihuih/llmservice/moe-mlm
run_cmd="
    cd $DIR && \
    ${PROFILE_CMD} python -u ${MEGATRON_PATH}/pretrain_gpt.py \
        ${MEGATRON_EXTRA_PARAMS} \
        ${MODEL_SPECIFIC_ARGS[@]} \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${MOE_ARGS[@]}
    "

# Container
# if [ "$PR" == "fp8" ]; then
# CONT=gitlab-master.nvidia.com/shiqingf/pytorch-docker:pytorch-24.01-te-1.2.1-MegablocksGroupedGEMM
# CONT=nvcr.io/nvidia/nemo:24.01.framework
# CONT=nvcr.io/nvidia/pytorch:24.01-py3
CONT=/lustre/fsw/coreai_dlalgo_llm/yihuih/images/24.01.sqsh
#CONT=gitlab-master.nvidia.com/denliu/dockers:moe-dev-eos-fp8
# else
#     CONT=gitlab-master.nvidia.com/denliu/dockers:moe-dev-eos
# fi

CONT_NAME=docker_pytorch
CONT_MOUNT=/lustre/:/lustre/

srun --jobid=360688 -N8 --ntasks-per-node 8 -pmix -l  \
    --container-name="${CONT_NAME}" \
    --container-image="${CONT}" \
    --container-mounts="${CONT_MOUNT},${DIR}:${DIR}" \
    --no-container-mount-home \
    bash -c "${run_cmd}"


# srun -pmix -l  \
#     --container-name="${CONT_NAME}" \
#     --container-image="${CONT}" \
#     --container-mounts="${CONT_MOUNT},${DIR}:${DIR}" \
#     --container-entrypoint --no-container-mount-home \
#     sh -c ${run_cmd} 2>&1 | tee -a ${LOGS_PATH}/${NAME}.log

#set -x
