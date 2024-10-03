#!/bin/bash

# Use: ./train.sh <data-path> <tokenizer-path>

# MODEL_SCALE="800M" # or "8B"
MODEL_SCALE="8B"

case "${MODEL_SCALE}" in
    "800M")
        TENSOR_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=48
        HIDDEN_SIZE=1024
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=32
        ;;
    "8B")
    # num_attention_heads (1) must be a multiple of tensor_model_parallel_size
        TENSOR_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=56
        HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=32
        GLOBAL_BATCH_SIZE=12
        ;;
    *)
        echo "Invalid version specified"
        exit 1
        ;;
esac

DATA_PATH=$1
TOKENIZER_PATH=$2

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

SEQ_LEN=4096
# TRAIN_SAMPLES=73242188  # 300B tokens / 4096
# LR_WARMUP_SAMPLES=50000
# LR_DECAY_SAMPLES=73192188 # TRAIN_SAMPLES - LR_WARMUP_SAMPLES

# TRAIN_SAMPLES=10000  # 300B tokens / 4096
# LR_WARMUP_SAMPLES=1000
# LR_DECAY_SAMPLES=9000 # TRAIN_SAMPLES - LR_WARMUP_SAMPLES

TRAIN_SAMPLES=10000  # 300B tokens / 4096
LR_WARMUP_SAMPLES=0
LR_DECAY_SAMPLES=1000 # TRAIN_SAMPLES - LR_WARMUP_SAMPLES



PP_SIZE=8
LR="5e-5"
MIN_LR="5e-6"
# OUT_DIR="training_decoder_10000/pp${PP_SIZE}_tp1/unfreeze_entire_decoder/soup-01_S_Q_A/trained_${TRAIN_SAMPLES}/batch_16/${LR}_clip0_3_wd0_3_warm10/"
OUT_DIR="training_decoder_1000/pp${PP_SIZE}_tp1/unfreeze_entire_decoder/D_01_Q_A/trained_${TRAIN_SAMPLES}/batch_16/${LR}_clip0_3_wd0_3_warm10/"

CHECKPOINT_DIR="/workspace/data/ssm-retrieval/mamba2-8b/checkpoints/${OUT_DIR}"
DATACACHE_DIR="./data-cache/${OUT_DIR}"
TENSORBOARD_DIR="./tensorboard/${OUT_DIR}"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

export TRITON_CACHE_DIR="./triton-cache/"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"




options=" \
       --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
       --sequence-parallel \
       --pipeline-model-parallel-size ${PP_SIZE} \
       --use-distributed-optimizer \
       --overlap-param-gather \
       --overlap-grad-reduce \
       --untie-embeddings-and-output-weights \
       --init-method-std 0.02 \
       --position-embedding-type none \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_ATTENTION_HEADS} \
       --num-query-groups 8 \
       --hybrid-attention-ratio 0 \
       --hybrid-mlp-ratio 0 \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --train-samples ${TRAIN_SAMPLES} \
       --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
       --lr-decay-samples ${LR_DECAY_SAMPLES} \
       --save ${CHECKPOINT_DIR} \
       --data-path ${DATA_PATH} \
       --data-cache-path ${DATACACHE_DIR} \
       --split 99,1,0 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --distributed-backend nccl \
       --micro-batch-size 2 \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --lr ${LR} \
       --min-lr ${MIN_LR} \
       --lr-decay-style cosine \
       --weight-decay 0.3 \
       --clip-grad 0.5 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10 \
       --save-interval 100 \
       --eval-interval 2000 \
       --eval-iters 2 \
       --bf16 \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
       --tensorboard-dir ${TENSORBOARD_DIR} \

       --pretrained-checkpoint  /workspace/data/ssm-retrieval/mamba2-8b/pp${PP_SIZE}_tp1 \
       --finetune \

       
        "
        # --inserting_mamba_states True \
        # --insert_mamba_states_for_training True \
        # --insert_mamba_states_for_training_dir /workspace/data/ssm-retrieval/data/hotpot/training_data/1000_valid_all/hidden_states/soup-01/ 
        # "
        
        
        # --insert_mamba_states_for_training_dir /workspace/data/ssm-retrieval/data/hotpot/training_data/100_valid_all/hidden_states/soup0-3/ 
        # "




        # --insert_mamba_states_for_training_dir /workspace/data/ssm-retrieval/data/hotpot/training_data/10000_valid_all/hidden_states/soup0-3/  
        # "

# --load ${CHECKPOINT_DIR} \
torchrun --nproc_per_node ${PP_SIZE} ../../pretrain_mamba.py ${options}
