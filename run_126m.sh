#!/bin/bash

#SBATCH -p luna 
#SBATCH -A dlfw 
#SBATCH -t 4:00:00 
#SBATCH -N 4 
#SBATCH --exclusive 
#SBATCH --mem=0 
#SBATCH --overcommit 
#SBATCH --ntasks-per-node=8 
#SBATCH --dependency=singleton 
#SBATCH -J dlfw-megatron:126m_bf16_baseline

NAME="126m_bf16_baseline"

DIR=`pwd`
WORKDIR="/workspace/gpt3"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

TENSORBOARD_DIR="${DIR}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

CHECKPOINT_DIR="${DIR}/checkpoints/${NAME}"
mkdir -p ${CHECKPOINT_DIR}

HISTOGRAM_DIR="${DIR}/histograms/${NAME}"
mkdir -p ${HISTOGRAM_DIR}

mounts="/lustre/fsw/adlr:/lustre/fsw/adlr,/lustre/fsw/dlfw:/lustre/fsw/dlfw,${HISTOGRAM_DIR}:/workspace/gpt3/histograms/cuda"

# Get the data blend
. /lustre/fsw/dlfw/dlfw-perf/ksivamani/megatron-fp8/data/gpt3/gpt3_blend.sh

freq="128"
#export LINEAR="{fi:{e:4,m:3,s:1,f:1,p:0,v:$freq},fw:{e:4,m:3,s:1,f:1,p:0,v:$freq},do:{e:4,m:3,s:1,f:1,p:0,v:$freq},fo:{v:$freq},di:{v:$freq},dw:{v:$freq}}"
export LINEAR="{fi:{v:$freq},fw:{v:$freq},do:{v:$freq},fo:{v:$freq},di:{v:$freq},dw:{v:$freq}}"

options=" \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --train-iters 2000 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --vocab-file /lustre/fsw/dlfw/dlfw-perf/ksivamani/megatron-fp8/data/bpe/gpt2-vocab.json \
    --merge-file /lustre/fsw/dlfw/dlfw-perf/ksivamani/megatron-fp8/data/bpe/gpt2-merges.txt \
    --save-interval 10000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.023 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR} "
    # --rampup-batch-size 32 32 1953125 \

run_cmd="${WORKDIR}/bind.sh --cpu=${WORKDIR}/dgxa100_ccx.sh --mem=${WORKDIR}/dgxa100_ccx.sh python -u ${WORKDIR}/pretrain_gpt.py ${options}"

srun -l \
     --container-image "gitlab-master.nvidia.com/ksivamani/containers:pytorch_21.10-py3_clippy" \
     --container-mounts "${mounts}" \
     --output=$DIR/logs/%x.log sh -c "${run_cmd}"

set +x
