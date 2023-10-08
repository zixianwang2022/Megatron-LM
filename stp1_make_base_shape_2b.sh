#!/bin/bash

#SBATCH -p interactive -A llmservice_nlp_fm -t 00:30:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=llmservice_nlp_fm-largelm:gpt3-muT-make-shape

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

NAME="gpt3-muT-make-shape"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

proxy_hidden_size=256

MUT_CONFIG_FILE=$DIR/muT_config/muT_config_2b.yaml
# SHAPE_FILE=$DIR/muT_config/proxy_shape_file_2b_hz2048_tp1.bsh
SHAPE_FILE=$DIR/muT_config/proxy_shape_file_2b_hz2048_tp1_squaredrelu.bsh



    # --num-layers 32 \
    # --hidden-size 4096 \
    # --num-attention-heads 32 \
    # --seq-length 4096 \
# --DDP-impl local

options=" \
    --muT_config_file $MUT_CONFIG_FILE \
    --shape_file $SHAPE_FILE \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --squared-relu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --hidden-size $proxy_hidden_size \
    --num-layers 24 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --train-samples 977550 \
    --lr-decay-samples 928673 \
    --lr-warmup-samples 296 \
    --lr 2e-4 \
    --min-lr 2e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16"

run_cmd="python -u ${DIR}/megatron_cal_shape.py ${options}"

srun -l \
     --container-image "/lustre/fsw/adlr/adlr-nlp/images/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels_pyspy.sqsh" \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr" \
     --no-container-mount-home \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

