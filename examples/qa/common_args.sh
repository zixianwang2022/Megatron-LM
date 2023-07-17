#! /bin/bash

. ./examples/qa/config.sh

if [[ $model_size == "843m" ]]; then
    mod_par=1
    layers=24
    hid_dim=1024
    heads=16
    pip_par=1
fi

if [[ $model_size == "2b" ]]; then
    mod_par=1
    layers=24
    hid_dim=2048
    heads=16
    pip_par=1
fi


if [[ $model_size == "8b" ]]; then
    mod_par=4
    layers=32
    hid_dim=4096
    heads=32
    pip_par=1
fi

if [[ $model_size == "22b" ]]; then
    mod_par=8
    layers=40
    hid_dim=6144
    heads=48
    pip_par=1
fi

if [[ $model_size == "43b" ]]; then
    mod_par=8
    layers=48
    hid_dim=8192
    heads=64
    pip_par=4
    if [[ $model_card == pp1 ]]; then
        pip_par=1
    fi
fi


## --use-flash-attn \ remove flash attention for reproducable results. 
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
        --seq-length 4096 \
        --max-position-embeddings 4096 \
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
        --DDP-impl local \
"
        # --weight-decay 1.0e-1
        # --lr-decay-iters 10000 \

if [[ $model_card == *-pp1* ]]; then
    GPT_ARGS+=" --use-distributed-optimizer"
fi

FT_ARGS="--data-folder $DATA_FOLDER \
    --eod-mask-loss \
    --answer-loss-only \
    --ft_neighbours ${ft_neighbours} \
    --task $TASK"

#if [[ $model_size == "43b" ]]; then
#    ARGS_43B="--num-layers-per-virtual-pipeline-stage 1 \
#    "
#    if [[ $model_card == *-pp1* ]]; then
#        ARGS_43B=""
#    fi
#    # --num-layers-per-virtual-pipeline-stage 1 " not yet supported
#    # --use-container-fused-kernels \
#    # DOCKER=/lustre/fsw/adlr/adlr-nlp/images/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels.sqsh
#    GPT_ARGS=" ${GPT_ARGS} ${ARGS_43B}"
#fi

DOCKER="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:22.04-py3-eval"
#DOCKER="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:22.12-py3-eval"