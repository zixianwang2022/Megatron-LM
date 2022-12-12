#!/bin/bash

######## Retro workdirs. ########
RETRO_WORKDIRS=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs

######## Data blend. ########
export BLEND_SCRIPT_DIR=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/misc

######## GPT, Bert. ########
export GPT_VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json
export GPT_MERGE_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt
export BERT_LOAD_PATH=/home/universal-lm-data-netapp/chkpts/bert/345M_no_rng
export BERT_VOCAB_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt

######## Pipeline. ########
export OLD_RETRO_WIKI_DB="/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5"
export OLD_RETRO_WIKI_DB_EMBED_DIR="/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/wiki-cls-indexes"
export OLD_RETRO_WIKI_INDEX="/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/Wikipedia_IVF262144_HNSW32_Flat_index.bin"
export OLD_RETRO_WIKI_TRAIN_DATA="/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.h5py"
export OLD_RETRO_WIKI_TRAIN_NBRS="/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.neighbors.wiki.hdf5"
export OLD_RETRO_WIKI_VALID_DATA="/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.h5py"
export OLD_RETRO_WIKI_VALID_NBRS="/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/wiki.valid.h5py_start_0_end_25600_ns_2037248_sl2048_seed_1234_with_offset.tokens.feat.h5py_neighbors.wiki.hdf5"
# export NEW_RETRO_WIKI_INDEX="/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/IVF262144_HNSW32,Flat/added_0667_0000-0666.faissindex"

# eof.
