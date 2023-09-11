#!/bin/bash
# bash examples/qa/finetune_normal_lm.sh landrover_tasb_retrieved 843m 1 3e-6 1 

model_size=70b
global_bsz=2
lr=1.0e-5
# model_card=llama2_text_70b_pp8_itp-32k
model_card=llama2_text_70b
# model_card=llama2_text_70b_itp-32k
# model_card=llama2_text_7b

TASK=None
train_iters=1000

. ./examples/long_context_flan_llama2/long_context_llama2_pretrain_args.sh

num_nodes=1
num_gpus=8

min_lr=0.000001
if [[ $model_size == "70b" ]]; then
    num_nodes=8
    min_lr=0.00000001
fi

SAVENAME="${model_card}_${model_size}_${global_bsz}_${lr}"
CHECKPOINT_PATH="${QA_HOME}/checkpoints/applications/${SAVENAME}"
TENSORBOARD_DIR="${QA_HOME}/tensorboard/${SAVENAME}"
mkdir -p ${TENSORBOARD_DIR}

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 500 \
             --eval-interval 2 \
             --tensorboard-dir ${TENSORBOARD_DIR} \
             --log-validation-ppl-to-tensorboard \
             --eval-iters 2"

options=" \
    $GPT_ARGS \
    --weight-decay 0.0 \
    --lr-decay-style constant \
    --sequence_parallel \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --data-path 1.0 /lustre/fsw/adlr/adlr-nlp/pengx/long_context_llm/megatron-lm/eval_data/bin-idx/pg19_test/pg19_test.llama2_text_document \
    --lr ${lr} \
    --lr-warmup-iters 20 \
    --lr-warmup-init 1e-6 \
    --init-method-std 0.010 \
    --split 99,1,0 \
    --micro-batch-size 1 \
    --global-batch-size ${global_bsz} \
    --train-iters ${train_iters} \
    --dataloader-type cyclic \
    --save $CHECKPOINT_PATH \
    $OUTPUT_ARGS \
    $FT_ARGS"

if [[ -f "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" ]]; then
    options="$options \
        --load $CHECKPOINT_PATH "
else
    options="$options \
        --load $PRETRAINED_CHECKPOINT \
        --finetune \
	--no-load-rng \
        --no-load-optim "
fi

DIR=`pwd`
# -m torch.distributed.launch --nproc_per_node 8
run_cmd="python -u  ${DIR}/pretrain_gpt.py ${options}"
# srun -l \
#      --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/faissgpu" \
#      --container-mounts "/home/pengx/projects/retro/:/home/pengx/projects/retro/" \
#      --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"
# $run_cmd

export SUBMIT_LOGS="${QA_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


echo ${run_cmd}
# ${run_cmd}
# export SUBMIT_ACCOUNT=llmservice_nlp_fm
submit_job --gpu ${num_gpus} --nodes ${num_nodes} --email_mode never  --mounts $MOUNTS --partition $PARTITION  --image $DOCKER -c "$LAUNCH ${run_cmd}" -n "${SAVENAME}" --duration 0.5  --exclude luna-0534,luna-0253,luna-0377 # --dependent_clones 5
