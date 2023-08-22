#!/bin/bash

#SBATCH -p batch_block1,batch_block2
#SBATCH --nodes=1
#SBATCH -A adlr
#SBATCH -t 2:00:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:llama
#SBATCH --ntasks-per-node=1

# ... SBATCH --dependency=singleton

set -u

######## Arguments. ########

. ${SCRIPT_DIR}/../args_gen.sh ${MODEL_FAMILY} ${MODEL_TYPE} ${MODEL_SIZE}

######## Task args. ########

ARGS=" \
    ${ARGS} \
    --top_k 1 \
    --top_p 0.0 "

TASK_OPTIONS=" \
   --models=${MODEL_FAMILY} \
   ${TASK_ARGS} \
   --undefok=sequence-parallel,recompute-activations,use-flash-attn,overlap-p2p-communication,apply-layernorm-1p,untie-embeddings-and-output-weights,disable-bias-linear,no-position-embedding,use-rotary-position-embeddings,rotary-percent,swiglu,attention-dropout,hidden-dropout,exit-duration-in-mins,tensor-model-parallel-size,pipeline-model-parallel-size,num-layers,hidden-size,num-attention-heads,seq-length,max-position-embeddings,micro-batch-size,global-batch-size,rampup-batch-size,train-samples,lr-decay-samples,lr-warmup-samples,lr,min-lr,lr-decay-style,log-interval,eval-iters,eval-interval,tokenizer-type,tokenizer-model,save-interval,load,split,clip-grad,weight-decay,adam-beta1,adam-beta2,init-method-std,log-params-norm,log-num-zeros-in-grad,bf16,DDP-impl,top_k,top_p,num-layers-per-virtual-pipeline-stage\
\
,norm-epsilon,no-masked-softmax-fusion,no-load-optim,no-load-rng,fp16,_model_family,_model_type,_model_size,norm-type,exit-on-missing-checkpoint,use-checkpoint-args,no-query-key-layer-scaling,use-llama-rotary-emb,use-llama-qkv,use-llama-mlp,use-llama-matmul,use-llama-default-dtype,group-query-attention,num-query-groups,log-world-size-to-tensorboard"

######## Command. ########

IMAGE=gitlab-master.nvidia.com/jupinderp/bigbench_containers:v1
MOUNT=/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2:/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2,/home/lmcafee/src/lutil:/home/lmcafee/src/lutil

srun \
    --container-image ${IMAGE} \
    --container-mounts ${MOUNT} \
    bash -c "
  export PYTHONPATH=${MEGATRON_REPO_DIR}:${LLAMA_REPO_DIR}:${BIG_BENCH_REPO_DIR}:/home/lmcafee/src;
  export CUDA_DEVICE_MAX_CONNECTIONS=1;
  export NCCL_IB_SL=1;
  pip install einops;
  pip install sacrebleu --upgrade;
  pip install fairscale;
  pip install -U transformers;
  set -x;
  python ${BIG_BENCH_REPO_DIR}/bigbench/evaluate_task.py ${ARGS} ${TASK_OPTIONS}
"

# eof.
