#! /bin/bash


. ./examples/long_context_flan/config.sh 

if [[ $model_size == "8b" ]]; then
    mod_par=4
    layers=32
    hid_dim=4096
    heads=32
    pip_par=1
fi

if [[ $model_size == "43b" ]]; then
    mod_par=8
    layers=48
    hid_dim=8192
    heads=64
    pip_par=4
    if [[ $model_card == *itp-32k* ]]; then
        mod_par=16
    fi
fi

GPT_ARGS="--apply-layernorm-1p \
	--use-flash-attn \
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
	--use-distributed-optimizer \
	--exit-duration-in-mins 230 \
        --DDP-impl local"
        # --weight-decay 1.0e-1
        # --lr-decay-iters 10000 \

if [[ ${model_card} == *itp-16k*  ]]; then
    GPT_ARGS="$GPT_ARGS \
    --seq-length 16384 \
    --max-position-embeddings 16384 \
    --rotary-seq-len-interpolation-factor 4 \
    --distributed-timeout-minutes 30"

elif [[ ${model_card} == *itp-32k*  ]]; then
    GPT_ARGS="$GPT_ARGS \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --recompute-activations \
    --rotary-seq-len-interpolation-factor 8 \
    --distributed-timeout-minutes 30"
else
    GPT_ARGS="$GPT_ARGS \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --distributed-timeout-minutes 30"
fi


FT_ARGS="--eod-mask-loss \
    --answer-loss-only \
    --task None"

DOCKER="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:22.04-py3-eval"
