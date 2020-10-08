#!/bin/bash

#SBATCH -p luna -A adlr-nlp -t 4:00:00 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_t5.py ${OPTIONS} ${CONFIG_ARGS}"

srun -l \
     --container-image "nvcr.io#nvidia/pytorch:20.06-py3" \
     --container-mounts "/lustre/fsw/adlr-nlp:/lustre/fsw/adlr-nlp" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x
