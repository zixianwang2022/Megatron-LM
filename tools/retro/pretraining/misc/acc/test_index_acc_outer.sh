#!/bin/bash

#SBATCH -p batch_dgx2h_m2 --gres=gpu:16 -A gpu_adlr_nlp -t 8:00:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retro

NAME="lmcafee-retro"

# DIR=$(readlink -f `pwd`)
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# LOG_DIR=$DIR/logs/acc
LOG_DIR=/home/lmcafee-src/megatrons/megatron-lm-boxin/lawrence/logs/acc/n2000
mkdir -p $LOG_DIR

# >>>
# task=proc
# task=vis
task=test-knn

# i0="OPQ64_128,IVF4194304_HNSW32,PQ64__t66630804"
index="OPQ64_128,IVF4194304_HNSW32,PQ64__t65191936"
# index="OPQ64_256,IVF4194304_HNSW32,PQ64__t66630804"
# index="OPQ64_512,IVF4194304_HNSW32,PQ64__t66630804"

# i1="OPQ64_128,IVF4096_HNSW32,PQ64__t10000"
# i1="OPQ64_64,IVF4194304_HNSW32,PQ64__t66630804"
# i1="OPQ32_128,IVF4194304_HNSW32,PQ32__t66630804"
# i1="OPQ32_64,IVF4194304_HNSW32,PQ32__t66630804"
# i1="OPQ16_128,IVF4194304_HNSW32,PQ16__t66630804"
# i1="Flat"
# <<<

# run_cmd="pwd && cd $DIR && pwd && bash lawrence/acc/test_index_acc_inner.sh $task $i0 $i1"
run_cmd="pwd && cd $DIR && pwd && bash lawrence/acc/test_index_acc_inner.sh $task $index"
CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee"
srun -l \
     --container-image "nvcr.io#nvidia/pytorch:22.04-py3" \
     --container-mounts $CONTAINER_MOUNTS \
     --output=$LOG_DIR/%j__${i1}.log sh -c "${run_cmd}"

set +x

# eof
