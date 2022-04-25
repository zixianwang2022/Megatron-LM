#!/bin/bash

#SBATCH -p luna -A adlr -t 04:00:00 --nodes=128 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=adlr-nlp:baseline.gpt3.bf16.175b

NAME="baseline.gpt3.bf16/175b"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

DIR="/lustre/fsw/gpu-comparch/jkamalu/fp8-175b/v2/${NAME}"

# branch: fp8-main, commit: 15f960749588dfda5b671a1e9b2cf9dbfc0d88ca
MEGATRON_DIR="/lustre/fsw/gpu-comparch/jkamalu/fp8-175b/v2/megatron-lm.v2"

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
EXIT_DURATION=$((timeleft[0]*60 + timeleft[1] - 20))
echo "setting exit duration to $EXIT_DURATION minutes"
################################################################################

options=" \
    --exit-duration-in-mins ${EXIT_DURATION} \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --rampup-batch-size 32 32 4882813 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 406902 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 10 \
    --eval-iters 50 \
    --eval-interval 1000 \
    --data-path ${DATA_BLEND} \
    --vocab-file ${BPE_DIR}/gpt2-vocab.json \
    --merge-file ${BPE_DIR}/gpt2-merges.txt \
    --save-interval 5000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --checkpoint-activations"

run_cmd="${MEGATRON_DIR}/bind.sh --cpu=${MEGATRON_DIR}/dgxa100_ccx.sh --mem=${MEGATRON_DIR}/dgxa100_ccx.sh python -u ${MEGATRON_DIR}/pretrain_gpt.py ${options}"

srun -l \
     --container-image "/lustre/fsw/adlr/adlr-nlp/images/pytorch+bf16_nccl_fusion+pyspy.sqsh" \
     --container-mounts "/lustre/fsw/gpu-comparch:/lustre/fsw/gpu-comparch,/lustre/fsw/adlr:/lustre/fsw/adlr" \
     --output=$LOG_DIR/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

