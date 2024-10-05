#!/bin/bash

MODEL_CKPT=$1

echo "Input Path: $MODEL_CKPT"

./run_text_gen_server_8b.sh \
        $MODEL_CKPT \
        /workspace/data/ssm-retrieval/mamba2-8b/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model 
