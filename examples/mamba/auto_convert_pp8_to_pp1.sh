#!/bin/bash

PP8_CKPT=$1


# Use sed to replace 'pp8_tp1' with 'pp1_tp1' and store it in another variable
PP1_CKPT=$(echo "$PP8_CKPT" | sed 's/pp8_tp1/pp1_tp1/')

# Now, PP1_CKPT holds the updated path, you can print it or use it later
echo "Input Path: $PP8_CKPT"
echo "Output Path: $PP1_CKPT"


pip install megatron.core

python ../../tools/checkpoint/hybrid_conversion.py \
        --load-dir $PP8_CKPT \
        --save-dir $PP1_CKPT \
        --target-tp-size 1 \
        --target-pp-size 1 \
        --megatron-path /workspace/megatron 


pip uninstall megatron.core