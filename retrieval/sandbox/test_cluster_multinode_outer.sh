#!/bin/bash

#SBATCH -p batch --nodes=2 --gres=gpu:8 -A gpu_adlr_nlp -t 0:30:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retrieval

# ....... SBATCH -p batch --gres=gpu:8 -A gpu_adlr_nlp -t 4:00:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retrieval

# >>>
# [selene] SBATCH -p luna -A adlr -t 4:00:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=1 --job-name=adlr-nlp-dev:retrieval
# [draco-rno] SBATCH -p batch_dgx2h_m2 --gres=gpu:16 -A gpu_adlr_nlp -t 8:00:00 --exclusive --nv-meta=ml-model:language-modeling --job-name=adlr-nlp-dev:retrieval
# [draco-rno] *batch_dgx2h_m2, batch_short_dgx2h_m2
# <<<

# DIR=$(readlink -f `pwd`)
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOG_DIR="./lawrence/sandbox/logs"
mkdir -p $LOG_DIR

# >>>
# ntrain=10000
# ntrain=100000
# ntrain=200000
# ntrain=500000
ntrain=1000000
# ntrain=20000000
# ntrain=50000000
# ntrain=100000000 # *
# ntrain=200000000
# ntrain=300000000
# ntrain=400000000

# ncenters=128
# ncenters=512
# ncenters=4096
# ncenters=8192
ncenters=16384
# ncenters=262144
# ncenters=4194304 # *
# <<<

# >>>
cmd="python -m lawrence.sandbox.test_cluster_multinode \
            --ncenters ${ncenters} \
            --ntrain ${ntrain} \
"
# cmd="ls -alh"
# <<<

# run_cmd="pwd && cd $DIR && pwd && bash lawrence/sandbox/test_cluster_multinode_inner.sh '$cmd'"
run_cmd="cd $DIR && bash lawrence/sandbox/test_cluster_multinode_inner.sh '$cmd'"

if [[ $HOSTNAME == *"rno"* ]]; then
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee,/gpfs/fs1/projects/gpu_adlr/datasets/boxinw:/gpfs/fs1/projects/gpu_adlr/datasets/boxinw"

elif [[ $HOSTNAME == *"luna-"* ]]; then
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/lustre/fsw/adlr:/lustre/fsw/adlr"

elif [[ $HOSTNAME == *"ip-"* ]]; then
    CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/mnt/fsx-outputs-chipdesign/lmcafee:/mnt/fsx-outputs-chipdesign/lmcafee"

else
    echo "error ... specialize for host '$HOSTNAME'."
    exit 1
fi

srun -l \
     --container-image "nvcr.io#nvidia/pytorch:22.04-py3" \
     --container-mounts $CONTAINER_MOUNTS \
     --output=$LOG_DIR/%j__${ncenters}__${ntrain}.log \
     sh -c "${run_cmd}"

set +x

# eof
