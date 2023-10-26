#!/bin/bash

srun -p interactive,batch_singlenode -A llmservice_nlp_fm -N 1 --pty \
     --container-image /lustre/fsw/portfolios/llmservice/users/jbarker/workspace/containers/adlr+megatron-lm+pytorch+23.04-py3-jbarker.sqsh \
     --container-mounts "/lustre" \
     --job-name "llmservice_nlp_fm-megatron-dev:interactive" \
     -t 2:00:00 \
     bash -l
