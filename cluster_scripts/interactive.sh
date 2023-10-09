#!/bin/bash

#--container-image nvcr.io#nvidia/pytorch:23.04-py3 \
srun -p interactive,luna -A llmservice_nlp_fm -N 1 --pty \
     --container-image /lustre/fsw/adlr/adlr-nlp/jbarker/checkpoints/adlr+megatron-lm+pytorch+23.04-py3-jbarker-revilm.sqsh \
     --container-mounts "/lustre/fsw/adlr/adlr-nlp:/lustre/fsw/adlr/adlr-nlp,/lustre/fsw/adlr/adlr-nlp-large:/lustre/fsw/adlr/adlr-nlp-large,/lustre/fsw/adlr/adlr-others:/lustre/fsw/adlr/adlr-others" \
     --job-name "llmservice_nlp_fm-megatron-dev:interactive" \
     -t 2:00:00 \
     bash -l
