#! /bin/bash


. ./examples/long_context_flan_llama2/config_llama2.sh 

## llama-2
if [[ $model_size == "7b" ]]; then
    mod_par=2
    layers=32
    hid_dim=4096
    heads=32
    pip_par=1
fi

if [[ $model_size == "13b" ]]; then
    mod_par=4
    layers=40
    hid_dim=5120
    heads=40
    pip_par=1
fi

if [[ $model_size == "70b" ]]; then
    mod_par=8
    layers=80
    hid_dim=8192
    heads=64
    pip_par=4
    if [[ $model_card == *pp8* ]]; then
        pip_par=8
    fi
fi


GPT_ARGS="--use-flash-attn \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --swiglu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --pipeline-model-parallel-size $pip_par \
        --tensor-model-parallel-size $mod_par \
        --num-layers $layers \
        --hidden-size $hid_dim \
        --num-attention-heads $heads \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model ${TOKENIZER_MODEL_LLAMA2} \
        --clip-grad 1.0 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --bf16 \
        --exit-duration-in-mins 220 \
        --exit-on-missing-checkpoint \
        --use-checkpoint-args \
        --normalization RMSNorm \
        --no-masked-softmax-fusion \
        --no-query-key-layer-scaling \
        --use-distributed-optimizer \
        --norm-epsilon 1e-05 "


if [[ ${model_card} == *itp-16k*  ]]; then
    GPT_ARGS="$GPT_ARGS \
    --seq-length 16384 \
    --max-position-embeddings 16384 \
    --max-tokens-to-oom 16384 \
    --rotary-seq-len-interpolation-factor 4 \
    --distributed-timeout-minutes 30"

elif [[ ${model_card} == *itp-32k*  ]]; then
    GPT_ARGS="$GPT_ARGS \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --max-tokens-to-oom 32768 \
    --rotary-seq-len-interpolation-factor 8 \
    --distributed-timeout-minutes 30"
else
    GPT_ARGS="$GPT_ARGS \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --distributed-timeout-minutes 30"
fi

if [[ ${model_card} == *text*70b* ]]; then
    PRETRAINED_CHECKPOINT=${llama2_text_70b}
    if [[ ${model_card} == *text*70b*pp8* ]]; then
        PRETRAINED_CHECKPOINT="/lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/llama2_megatron_text_70b_pp8/"
    fi
fi
if [[ ${model_card} == *text*7b* ]]; then
    PRETRAINED_CHECKPOINT=${llama2_text_7b}
fi

FT_ARGS="--eod-mask-loss"

if [[ ${model_card} == *itp*  ]]; then
    FT_ARGS="$FT_ARGS \
    --recompute-method uniform \
    --recompute-granularity full"
    # --recompute-activations"
fi

DOCKER="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:22.04-py3-eval"
