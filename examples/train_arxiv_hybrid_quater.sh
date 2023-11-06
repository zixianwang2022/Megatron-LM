#!/bin/bash
#SBATCH -A llmservice_nlp_fm
#SBATCH -p batch_block1,batch_block2
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=8
#SBATCH --job-name=adlr-nlp-develop:arxiv-hybrid-8nodes-newloader-COCO2014-oci-flamingo-2b-1e-4

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SEQ_LEN=1024

NAME="flamingo-2b-aug-1e-4-coco-megatron-hybrid-arxiv-quater"
LOAD_NAME="gpt3-2b-multi-1.1t-gtc"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs


FINETUNE_DIR="/lustre/fsw/portfolios/llmservice/users/guilinl/checkpoint/megatron/${NAME}"
CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/zhuoliny/new-nvllm/${LOAD_NAME}"

TENSORBOARD_DIR="$DIR/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

DATA_TRAIN="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/_obsolete/yml_collection/ocr_arxiv.yaml"
DATA_VALID="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/_obsolete/yml_collection/ocr_arxiv.yaml"

VISUAL_ARCH="HybridSAMCLIP"

VISUAL_ARCH_SAM="SAM_L"
VISUAL_TYPE_SAM="sam"
VISUAL_LOAD_DIR_SAM="/lustre/fsw/portfolios/llmservice/users/guilinl/checkpoint/vision/SAM_L_selene"
VISUAL_SAVE_DIR_SAM="${FINETUNE_DIR}/${VISUAL_TYPE}"
IMG_H_SAM=1024
IMG_W_SAM=1024

VISUAL_ARCH_CLIP="L_14"
VISUAL_TYPE_CLIP="vit"
VISUAL_LOAD_DIR_CLIP="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/checkpoints/vit_L_14_336px"
VISUAL_SAVE_DIR_CLIP="${FINETUNE_DIR}/${VISUAL_TYPE}"
IMG_H_CLIP=336
IMG_W_CLIP=336

PROMPT_PATH="${DIR}/GPT4-prompts-augment.json"
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
    --train-samples 410000 \
    --micro-batch-size 2 \
    --global-batch-size 256 \
    --lr-decay-samples 25600000 \
    --lr-warmup-samples 83200 \
    --lr 1e-4 \
    --min-lr 5e-5 \
    --lr-decay-style cosine \
    --log-interval 10 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/users/zhuoliny/new-nvllm/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_TRAIN} \
    --valid-path ${DATA_VALID} \
    --prompt-path ${PROMPT_PATH} \
    --dset-config ${DATASET_CONFIG} \
    --save-interval 1000 \
    --save ${FINETUNE_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 50.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --add-gated-xattn \
    --add-BOS \
    --use-hybrid-visual-backbones \
    --xattn-sam-num 1 \
    --xattn-clip-num 3 \
    --visual-arch ${VISUAL_ARCH} \
    --visual-arch-sam ${VISUAL_ARCH_SAM} \
    --visual-path-sam ${VISUAL_LOAD_DIR_SAM} \
    --visual-type-sam ${VISUAL_TYPE_SAM} \
    --img-h-sam ${IMG_H_SAM} \
    --img-w-sam ${IMG_W_SAM} \
    --visual-arch-clip ${VISUAL_ARCH_CLIP} \
    --visual-path-clip ${VISUAL_LOAD_DIR_CLIP} \
    --visual-type-clip ${VISUAL_TYPE_CLIP} \
    --img-h-clip ${IMG_H_CLIP} \
    --img-w-clip ${IMG_W_CLIP} \
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
    --dataset-type nvgpt4 \
    --SAM-randinit \
    --align-to-old \
    --tensorboard-dir ${TENSORBOARD_DIR}"


run_cmd="python -u ${DIR}/pretrain_flamingo.py ${options}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`


srun -l \
     --container-image "/lustre/fsw/portfolios/llmservice/users/guilinl/docker/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels_pyspy.sqsh" \
     --container-mounts "/lustre/:/lustre/,/home/guilinl:/home/guilinl" \
     --output=$DIR/logs/%x_%j_$DATETIME.log \
     sh -c "${run_cmd}"

set +x

# python -u ${DIR}/pretrain_flamingo.py ${options}
