#!/bin/bash

#SBATCH -p interactive -A llmservice_nlp_fm -t 00:30:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=llmservice_nlp_fm-largelm:gpt3-eng-cc202240-1.1t-gtc-llr-muT-proxymodel-2b-lrLEARNING_RATE-initINIT_SCALE-embEMB_MULTIPLIER-baseshapeBASE_SHAPE_HZ_bzrampup

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

NAME="gpt3-eng-cc202240-1.1t-gtc-llr-muT-proxymodel-2b-lrLEARNING_RATE-initINIT_SCALE-embEMB_MULTIPLIER-baseshapeBASE_SHAPE_HZ_bzrampup"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/dasu/checkpoints/gpt3/gpt3-next-llm-eng-cc202240-1.1t-gtc/${NAME}"
BLENDABLE_INDEX="/lustre/fsw/adlr/adlr-nlp/dasu/data/blendable-index/${NAME}"

TENSORBOARD_DIR="$DIR/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}
mkdir -p $DIR/logs/muT_baseBASE_SHAPE_HZ

# SHAPE_FILE=$DIR/muT_config/proxy_shape_file_for8B.bsh
SHAPE_FILE=$DIR/muT_config/proxy_shape_file_2b_hz2048_tp1.bsh

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp/dasu/data/eng/eng-1.1t-gtc/eng-cc202240-1.1t-gtc-blend-v0.1.sh
    
    # --rampup-batch-size 32 32 65324160 \
    # --train-samples 268554688 \
    # --lr-decay-samples 255126953 \
    # --lr-warmup-samples 81381 \
    # --hidden-size 1024 \
    # --no-query-key-layer-scaling \
    # --lr-warmup-samples 21032 \
# --DDP-impl local \
proxy_hidden_size=256

DATA_CACHE_PATH=/lustre/fsw/adlr/adlr-nlp/dasu/data/eng/eng-1.1t-gtc/CC-MAIN-2022-40
## will use 4B tokens for the experiments. so the train samples will be 4/284*69406110=977550
options=" \
    --rampup-batch-size 64 64 244388 \
    --initialization-scale INIT_SCALE \
    --embedding-multiplier EMB_MULTIPLIER \
    --lr LEARNING_RATE \
    --min-lr MIN_LR \
    --shape_file $SHAPE_FILE \
    --sequence-parallel \
    --recompute-activations \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size $proxy_hidden_size \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --train-samples 977550 \
    --lr-decay-samples 928673 \
    --lr-warmup-samples 296 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --data-cache-path ${DATA_CACHE_PATH} \
    --save-interval 5000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --tensorboard-dir ${TENSORBOARD_DIR}"

run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_gpt.py ${options}"

srun -l \
     --container-image "/lustre/fsw/adlr/adlr-nlp/images/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels_pyspy.sqsh" \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr" \
     --output=$DIR/logs/muT_baseBASE_SHAPE_HZ/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

