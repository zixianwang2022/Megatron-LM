#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SEQ_LEN=256

NAME="flamingo-2b-debug-interactive"
LOAD_NAME="gpt3-2b-multi-1.1t-gtc"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

FINETUNE_DIR="/lustre/fsw/adlr/adlr-nlp/zhuoliny/checkpoints/flamingo-checkpoints/${NAME}"
CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/${LOAD_NAME}"

TENSORBOARD_DIR="$DIR/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

DATA_TRAIN="1.0000 /lustre/fsw/adlr/adlr-nlp/zhuoliny/debug_folder/COCO_train_mmdata_512sl_256k_vocab"
DATA_VALID="1.0000 /lustre/fsw/adlr/adlr-nlp/zhuoliny/debug_folder/COCO_val_mmdata_512sl_256k_vocab_mmdata"

VISUAL_ARCH="SAM_L"
VISUAL_TYPE="sam"
VISUAL_LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/zhuoliny/checkpoints/SAM_L_16"
VISUAL_SAVE_DIR="${FINETUNE_DIR}/${VISUAL_TYPE}"

PROMPT_PATH="${DIR}/GPT4-prompts.json"
DATASET_CONFIG="${DIR}/dataset.yaml"

options=" \
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
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length ${SEQ_LEN} \
    --ds-seq-length 512 \
    --max-position-embeddings 4096 \
    --cyclic-train-iters 100000000 \
    --train-samples 131072 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --lr-decay-samples 25600000 \
    --lr-warmup-samples 83200 \
    --lr 2.0e-5 \
    --min-lr 2.0e-6 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_TRAIN} \
    --valid-path ${DATA_VALID} \
    --prompt-path ${PROMPT_PATH} \
    --dset-config ${DATASET_CONFIG} \
    --save-interval 1000 \
    --save ${FINETUNE_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --add-gated-xattn \
    --add-BOS \
    --visual-arch ${VISUAL_ARCH} \
    --visual-path ${VISUAL_LOAD_DIR} \
    --visual-type ${VISUAL_TYPE} \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --no-load-optim \
    --eod-mask-loss \
    --finetune \
    --perceiver-type none \
    --freeze-LM \
    --freeze-ViT \
    --img-h 1024 \
    --img-w 1024 \
    --dataloader-type cyclic --no-data-sharding \
    --tensorboard-dir ${TENSORBOARD_DIR}"

python -u ${DIR}/pretrain_flamingo.py ${options}

