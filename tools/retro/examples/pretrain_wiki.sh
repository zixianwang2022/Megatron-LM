#!/bin/bash

set -u

. /gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/preprocess/gpt3_blend_wiki.sh
DATA_PATH=${DATA_BLEND}

# BPE_DIR="/lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/bpe"
VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-vocab.json
MERGE_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/mpatwary/checkpoints/gpt3/gpt3-357m/gpt2-merges.txt

# --neighbors_path /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.neighbors.wiki.hdf5 \
# --database_path /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5 \
# --valid_neighbors_path /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.feat.h5py_neighbors.wiki.hdf5 \
# --valid_database_path /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retrieval-gpt3/Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5 \
# --tensorboard-dir ${TENSORBOARD_DIR}
# --weight 1
# --log-validation-ppl-to-tensorboard
# --data-path2 ${DATA_BLEND} \
# --vocab-file ${BPE_DIR}/gpt2-vocab.json \
# --merge-file ${BPE_DIR}/gpt2-merges.txt \
# --save ${CHECKPOINT_DIR} \
# --load ${CHECKPOINT_DIR} \
# --exit-duration-in-mins 230 \
# --debug \
NUM_LAYERS=12 # 4, *12
HIDDEN_SIZE=512 # 256, 512, *768
NUM_HEADS=4 # 4, 8, *12
# --retro-add-retriever \
options=" \
    --retro-cyclic-train-iters 750000 \

    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 8 \
    --global-batch-size 256 \
    --train-samples  2037248  \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 100 \
    --eval-interval 2000 \
    --data-path ${DATA_PATH} \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --save-interval 10000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.023 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --fp16 \
    --loss-scale 1024 \
    --DDP-impl local \
    --dataloader-type cyclic \
    --no-data-sharding \
"

unset NCCL_DEBUG

NPROCS=1
# NPROCS=16
# SCRIPT=pretrain_gpt_retro.py
SCRIPT=pretrain_gpt.py
python -m torch.distributed.launch \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
    ${SCRIPT} \
    ${options} \
