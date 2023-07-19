#!/bin/bash

#SBATCH -p batch_block1,batch_block2
#SBATCH --nodes=4
#SBATCH -A adlr
#SBATCH -t 0:30:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:retro-nextlm
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton







# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# customize / begin.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

ADD_RETRIEVER=1
REPO_DIR="/home/boxinw/megatron-lm-pretrain"
CHECKPOINT_DIR="/lustre/fs1/portfolios/adlr/users/boxinw/retro/workdirs/next-llm/finetune/pretrain-checkpoint"

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# customize / end.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<







######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

DIR=$(readlink -f `pwd`)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOG_DIR=$DIR/logs
mkdir -p $LOG_DIR

######## checkpoint. ########

# TENSORBOARD_DIR="$CHECKPOINT_DIR/tensorboard"
# mkdir -p ${TENSORBOARD_DIR}

######## data blend. ########

# BLENDABLE_INDEX="???"
# --blendable-index-path ${BLENDABLE_INDEX} \

. /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/lawrence_blend_oci.sh

######## args. ########

# --save-interval 2000 \
# --save ${CHECKPOINT_DIR} \
# --unsafe-flag-for-gpt-speedup \
# --use-container-fused-kernels \
# --overlap-p2p-communication \
#     --load ${CHECKPOINT_DIR} \
#     --tensorboard-dir ${TENSORBOARD_DIR} \
#     \
TP=8
ARGS=" \
    --sequence-parallel \
    --recompute-activations \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 220 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 4 \
    --num-layers-per-virtual-pipeline-stage 1 \
    --num-layers 48 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 768 \
    --rampup-batch-size 192 192 65323968 \
    --train-samples 25000000 \
    --lr-decay-samples 23750000 \
    --lr-warmup-samples 16667 \
    --lr 9.0e-5 \
    --min-lr 9.0e-6 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fs1/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.007 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
"

######## retro. ########

if [ "$ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
else
    RETRO_WORKDIR=/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/next-llm
    ARGS="${ARGS} \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    "
    SCRIPT=pretrain_retro.py
fi

######## Command. ########

CMD=" \
    cd ${REPO_DIR} && \
    export PYTHONPATH=$PYTHONPATH:${REPO_DIR}:/home/lmcafee/src && \
    python -u ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo $CMD
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

IMAGE="gitlab-master.nvidia.com/lmcafee/sandbox-cluster/retro-process-22.12"
# IMAGE="/lustre/fsw/adlr/adlr-nlp/images/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels.sqsh"
MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/lustre/fsw/adlr/adlr-nlp/lmcafee:/lustre/fsw/adlr/adlr-nlp/lmcafee,/lustre/fs1/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t:/lustre/fs1/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t"
srun -l \
     --container-image $IMAGE \
     --container-mounts $MOUNTS \
     --output=$LOG_DIR/"%j_r${ADD_RETRIEVER}.log" \
     sh -c "${CMD}"

# eof.
