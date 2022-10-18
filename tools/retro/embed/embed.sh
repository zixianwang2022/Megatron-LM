#!/bin/bash

pip install h5py
pip install transformers

# cd /home/lmcafee-src/megatrons/megatron-lm-boxin

# python -m \
#     generation.generate_embeddings_bert_hdf5 \
#     --device 0 \
#     --input /gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/sampled_pretraining_corpus.chunks.hdf5 \
#     --output /gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/bert/0000.feat.hdf5 \
#     --bs 128 \
#     --split 16 \
#     --pointer 0

# echo "SHARE_SOURCE = $SHARE_SOURCE"
# exit 0

if [[ $HOSTNAME == *"draco-rno"* ]]; then
    python -m \
	   generation.generate_embeddings_bert_hdf5 \
	   --device 0 \
	   --input /gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/sampled_pretraining_corpus.chunks.hdf5 \
	   --output /gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/bert/0000.feat.hdf5 \
	   --split 80 \
	   --pointer 0

elif [[ $HOSTNAME == *"luna-"* ]]; then
    python -m \
	   generation.generate_embeddings_bert_hdf5 \
	   --device 0 \
	   --input /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/sampled_pretraining/sampled_pretraining_corpus.chunks.hdf5 \
	   --output /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/bert/0000.feat.hdf5 \
	   --split 80 \
	   --pointer 0

else
    echo "error ... specialize for host '$HOSTNAME'."
    exit 1
fi

# eof
