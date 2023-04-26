#!/bin/bash

#SBATCH -p gtc -A gtc -t 08:00:00 --nodes=32 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=gtc-nlp:22b

NAME="22b"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

DIR="/lustre/fsw/gpu-comparch/dstosic/fp8-22b/${NAME}"

# branch: fp8-main, commit: 15f960749588dfda5b671a1e9b2cf9dbfc0d88ca
MEGATRON_DIR="/lustre/fsw/gpu-comparch/dstosic/fp8-22b/megatron-lm.v2"

LOG_DIR="${DIR}/logs"
CHECKPOINT_DIR="${DIR}/checkpoints"
TENSORBOARD_DIR="${DIR}/tensorboard"

mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp-large/data/gpt3/gpt3_blend.sh

BPE_DIR="/lustre/fsw/adlr/adlr-nlp-large/data/bpe"

################################################################################
### Set exit duration based on variable time allocated for this specific job ###
# Query Slurm for the remaining job time left in the format [days-]hh:mm:ss
# format and pass the time (in units of minutes) to Megatron using variable
# EXIT_DURATION. The actual value passed is actually 13 minutes less for time
# to save model and extra margin. For our purposes we assume the days field
# will never be present to make parsing in bash easier. Note that setting
# EXIT_DURATION to 0 will terminate the job after 1 iteration.
timeleft=`squeue -j ${SLURM_JOBID} --noheader --format=%L`
timeleft=(`echo $timeleft | tr ':' ' '`)
EXIT_DURATION=$((timeleft[0]*60 + timeleft[1] - 10))
echo "setting exit duration to $EXIT_DURATION minutes"


options=" \
    --exit-duration-in-mins ${EXIT_DURATION} \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 48 \
    --hidden-size 6144 \
    --num-attention-heads 64 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --rampup-batch-size 32 32 3906250 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 325521 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --vocab-file /lustre/fsw/adlr/adlr-nlp-large/data/bpe/gpt2-vocab.json \
    --merge-file /lustre/fsw/adlr/adlr-nlp-large/data/bpe/gpt2-merges.txt \
    --save-interval 10000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.008 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --checkpoint-activations"

run_cmd="${MEGATRON_DIR}/bind.sh --cpu=${MEGATRON_DIR}/dgxa100_ccx.sh --mem=${MEGATRON_DIR}/dgxa100_ccx.sh python -u ${MEGATRON_DIR}/pretrain_gpt.py ${options}"

srun -l \
     --container-image "/lustre/fsw/adlr/adlr-nlp/images/pytorch+bf16_nccl_fusion.sqsh" \
     --container-mounts "/lustre/fsw/gpu-comparch:/lustre/fsw/gpu-comparch,/lustre/fsw/adlr:/lustre/fsw/adlr" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x
