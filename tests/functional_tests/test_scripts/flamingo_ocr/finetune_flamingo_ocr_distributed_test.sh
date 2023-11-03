#! /bin/bash
echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
   echo "$KEY=$VALUE"
done
echo "---------------------------------"

set -x
if [[ -n $MBS ]]; then MBS=1; fi
if [[ -n $GBS ]]; then GBS=8; fi

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
command="export CUDA_DEVICE_MAX_CONNECTIONS=1;"

TRANSFORMER_IMPL=local
TRAINING_DTYPE=bf16

if [[ $USE_CORE -eq 1 ]]; then
       echo "Running using megatron core"
       TRANSFORMER_IMPL=local
       TRAINING_DTYPE=bf16
       command="$command export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0;"
       USE_MCORE=1
       export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
fi

if [[ $USE_TE -eq 1 ]]; then
       echo "Running with TransformerEngine ..."
       TRANSFORMER_IMPL=transformer_engine
       TRAINING_DTYPE=bf16
else
       echo "Running with local transformer implementation ..."
fi
set +x

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"

torch_run_cmd="torchrun $DISTRIBUTED_ARGS \
    pretrain_flamingo.py \
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
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 96 \
    --ds-seq-length 512 \
    --max-position-embeddings 4096 \
    --micro-batch-size ${MBS:-4} \
    --global-batch-size ${GBS:-32} \
    --cyclic-train-iters $MAX_STEPS \
    --train-iters $MAX_STEPS \
    --timing-log-level 2 \
    --lr 1e-4 \
    --min-lr 5e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --save-interval 50 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model ${DATA_PATH}/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_PATH}/ocr_ft.yaml \
    --valid-path ${DATA_PATH}/ocr_ft.yaml \
    --prompt-path GPT4-prompts.json \
    --dset-config dataset.yaml \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --add-gated-xattn \
    --add-BOS \
    --visual-arch "SAM_L" \
    --visual-type "sam" \
    --visual-path ${DATA_PATH}/SAM_L_16 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --${TRAINING_DTYPE} \
    --DDP-impl local \
    --no-load-optim \
    --eod-mask-loss \
    --finetune \
    --perceiver-type none \
    --freeze-LM \
    --freeze-ViT \
    --img-h 1024 \
    --img-w 1024 \
    --dataloader-type cyclic --no-data-sharding \
    --SAM-randinit \
    --align-to-old \
    --transformer-impl $TRANSFORMER_IMPL \
    --dataset-type nvgpt4 \
    ${USE_MCORE:+--use-mcore-models} \
    --tensorboard-dir ${TENSORBOARD_DIR}"

command="$command $torch_run_cmd"
echo "-------------------- THE FINAL FINETUNE SCRIPT COMMAND THAT WILL BE RUN ------------"
echo "$command"
echo "------------------------------------------------------------------------------"

echo "$command" > $SCRIPTS_DIR/finetune_flamingo_distributed_test.sh
eval $command
