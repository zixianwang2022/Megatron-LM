#!/bin/bash

# 0000-0010,0020-0159

# ~~~~~~~~ corpus ~~~~~~~~
# for i in {000..159}; do

#     echo "... transfer $i ..."
#     # scp /gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/sampled_pretraining_corpus.chunks.hdf5.0${i}.feat.hdf5 selene-login:/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval/sampled_pretraining/

#     # ~~ corpus ~~
#     # scp draco-rno-dc-0001:/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/sampled_pretraining_corpus.chunks.hdf5.0${i}.feat.hdf5 /mnt/fsx-outputs-chipdesign/lmcafee/retrieval/corpus
#     rsync -avP draco-rno-dc-0001:/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/sampled_pretraining_corpus.chunks.hdf5.0${i}.feat.hdf5 /mnt/fsx-outputs-chipdesign/lmcafee/retrieval/corpus
#     # fpsync -vv -n 40 -o "-ravL --progress" draco-rno-dc-0001:/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/sampled_pretraining_corpus.chunks.hdf5.0${i}.feat.hdf5 /mnt/fsx-outputs-chipdesign/lmcafee/retrieval/corpus

# done

# ~~~~~~~~ wiki ~~~~~~~~
rsync -avP --append-verify --inplace draco-rno-dc-0001:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/data/rand-1m/* /mnt/fsx-outputs-chipdesign/lmcafee/retrieval/data/rand-1m/




# rsync -avP --append-verify --inplace draco-rno-dc-0001:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/v2/data1/pretraining_corpus.chunks.hdf5 /mnt/fsx-outputs-chipdesign/lmcafee/retrieval/wiki/
# rsync -avP --append-verify --inplace draco-rno-dc-0001:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/v2/data1/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.doc_ids.pkl /mnt/fsx-outputs-chipdesign/lmcafee/retrieval/wiki/

# for i in {00..15}; do
# for i in {12..15}; do

#     # ~~ corpus ~~
#     rsync -avP --append-verify --inplace draco-rno-dc-0001:/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/v2/data1/feat/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.00${i}.feat.hdf5 /mnt/fsx-outputs-chipdesign/lmcafee/retrieval/wiki/feat/

# done

# sampled_pretraining_corpus.chunks.hdf5

# eof
