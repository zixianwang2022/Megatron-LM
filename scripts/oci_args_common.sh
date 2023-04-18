#!/bin/bash

# >>>
# echo "${DATA_BLEND}"
# exit 0
# <<<

# --seed 1234 \
# --blendable-index-path ${BLENDABLE_INDEX} \

# --use-flash-attn \
# --use-container-fused-kernels \
# [x] --tokenizer-type GPTSentencePieceTokenizer \
# [x] --tokenizer-model /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
# [x] --finetune \
# [x] --retro-gpt-vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/gpt2-vocab.json \
# [x] --retro-gpt-merge-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/gpt2-merges.txt \
# ARGS=" \
#     --unsafe-flag-for-gpt-speedup \
#     --overlap-p2p-communication \
#     --apply-layernorm-1p \
#     --untie-embeddings-and-output-weights \
#     --disable-bias-linear \
#     --no-position-embedding \
#     --use-rotary-position-embeddings \
#     --rotary-percent 0.5 \
#     --swiglu \
#     --attention-dropout 0.0 \
#     --hidden-dropout 0.0 \
#     \
# --data-path 0.00759352333 /lustre/fs1/portfolios/adlr/users/lmcafee/retro/data/Reddit-Plus/Reddit_all_dialogue_shuf_text_document \
# ARGS=" \
#     --distributed-timeout-minutes 600 \
#     --tensor-model-parallel-size 1 \
#     --pipeline-model-parallel-size 1 \
#     --num-layers 24 \
#     --hidden-size 2048 \
#     --num-attention-heads 16 \
#     --micro-batch-size 1 \
#     --global-batch-size 256 \
#     --seq-length 4096 \
#     --max-position-embeddings 4096 \
#     --train-samples 24414063 \
#     --load /lustre/fs1/portfolios/adlr/users/lmcafee/bert-23/checkpoints \
#     --exit-on-missing-checkpoint \
#     --no-load-optim \
#     --data-path ${DATA_BLEND} \
#     --tokenizer-type BertWordPieceLowerCase \
#     --vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
#     --data-impl mmap \
#     --split 98,2,0 \
#     --distributed-backend nccl \
#     --lr 0.0001 \
#     --lr-decay-style linear \
#     --min-lr 1.0e-5 \
#     --lr-decay-samples 21158854 \
#     --lr-warmup-samples 15522 \
#     --weight-decay 1e-2 \
#     --clip-grad 1.0 \
#     --eval-interval 2000 \
#     --eval-iters 50 \
#     --bf16 \
#     --DDP-impl local \
#     --dataloader-type cyclic \
#     --no-data-sharding \
#     --no-gradient-accumulation-fusion \
#     --no-async-tensor-model-parallel-allreduce \
#     --bert-embedder-type megatron \
#     --output-bert-embeddings \
#     --retro-gpt-tokenizer-type GPTSentencePieceTokenizer \
#     --retro-gpt-tokenizer-model /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
#     --retro-gpt-seq-length 2048 \
#     --retro-gpt-chunk-length 64 \
#     --retro-bert-vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
#     --retro-bert-tokenizer-type BertWordPieceLowerCase \
#     --retro-tasks ${RETRO_TASKS} \
#     --retro-index-str ${RETRO_INDEX_STR} \
#     --retro-ef-search 4 \
#     --retro-nprobe 64 \
#     --retro-workdir ${RETRO_WORKDIR} \
#     --retro-nchunks-sampled 600000000 \
#     --retro-index-train-load-fraction ${RETRO_INDEX_TRAIN_LOAD_FRACTION} \
#     --retro-index-add-load-fraction ${RETRO_INDEX_ADD_LOAD_FRACTION} \
#     --retro-return-doc-ids \
#     --retro-no-delete-index-training-embeddings \
# "
# --bf16 \
ARGS=" \
    --distributed-timeout-minutes 600 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 16 \
    --global-batch-size 1024 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load /lustre/fs1/portfolios/adlr/users/lmcafee/bert-23/checkpoints \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --data-path ${DATA_BLEND} \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
    --data-impl mmap \
    --split ${RETRO_GPT_DATA_SPLIT} \
    --distributed-backend nccl \
    --lr 0.0001 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --train-samples ${RETRO_GPT_TRAIN_SAMPLES} \
    --lr-decay-samples ${RETRO_GPT_LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${RETRO_GPT_LR_WARMUP_SAMPLES} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --eval-interval ${RETRO_GPT_EVAL_INTERVAL} \
    --eval-iters ${RETRO_GPT_EVAL_ITERS} \
    --fp16 \
    --DDP-impl local \
    --dataloader-type ${DATALOADER_TYPE} \
    --no-data-sharding \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --bert-embedder-type megatron \
    --output-bert-embeddings \
    \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-tasks ${RETRO_TASKS} \
    --retro-bert-vocab-file /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
    --retro-bert-tokenizer-type BertWordPieceLowerCase \
    --retro-gpt-tokenizer-type GPTSentencePieceTokenizer \
    --retro-gpt-tokenizer-model /lustre/fs1/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --retro-gpt-hidden-size ${RETRO_GPT_HIDDEN_SIZE} \
    --retro-gpt-seq-length ${RETRO_GPT_SEQ_LENGTH} \
    --retro-gpt-global-batch-size ${RETRO_GPT_GLOBAL_BATCH_SIZE} \
    --retro-gpt-chunk-length ${RETRO_GPT_CHUNK_LENGTH} \
    --retro-gpt-return-doc-ids \
    --retro-index-str ${RETRO_INDEX_STR} \
    --retro-index-ntrain ${RETRO_INDEX_TRAIN_SAMPLES} \
    --retro-index-train-load-fraction ${RETRO_INDEX_TRAIN_LOAD_FRACTION} \
    --retro-index-add-load-fraction ${RETRO_INDEX_ADD_LOAD_FRACTION} \
    --retro-index-no-delete-training-embeddings \
    --retro-index-no-delete-added-codes \
    --retro-query-num-neighbors-query ${RETRO_QUERY_NUM_NEIGHBORS_QUERY} \
    --retro-query-num-neighbors-save ${RETRO_QUERY_NUM_NEIGHBORS_SAVE} \
    --retro-query-ef-search ${RETRO_QUERY_EF_SEARCH} \
    --retro-query-nprobe ${RETRO_QUERY_NPROBE} \
"

# eof.
