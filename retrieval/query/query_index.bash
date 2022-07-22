#!/bin/bash

# lawrence mcafee

set -u

DATA_DIR=/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval

# NTRAIN=66630804; NCLUSTER=4194304
NTRAIN=65191936; NCLUSTER=4194304
# NTRAIN=10000; NCLUSTER=4096
# INDEX_STR="Flat"

# INDEX_STR="OPQ64_64,IVF${NCLUSTER}_HNSW32,PQ64"
# INDEX_STR="OPQ64_128,IVF${NCLUSTER}_HNSW32,PQ64"
# INDEX_STR="OPQ64_256,IVF${NCLUSTER}_HNSW32,PQ64"
# INDEX_STR="OPQ64_512,IVF${NCLUSTER}_HNSW32,PQ64"

# INDEX_STR="OPQ32_64,IVF${NCLUSTER}_HNSW32,PQ32"
# INDEX_STR="OPQ32_128,IVF${NCLUSTER}_HNSW32,PQ32"
# INDEX_STR="OPQ32_256,IVF${NCLUSTER}_HNSW32,PQ32"
# INDEX_STR="OPQ32_512,IVF${NCLUSTER}_HNSW32,PQ32"

# INDEX_STR="OPQ16_128,IVF${NCLUSTER}_HNSW32,PQ16"
# INDEX_STR="OPQ16_256,IVF${NCLUSTER}_HNSW32,PQ16"
INDEX_STR="OPQ16_512,IVF${NCLUSTER}_HNSW32,PQ16"

INDEX_PATH="$DATA_DIR/index/${INDEX_STR}__t${NTRAIN}__added.faissindex"
OUT_DIR="$DATA_DIR/v2/n2000/${INDEX_STR}__t${NTRAIN}"
LOG_DIR="/home/lmcafee-src/megatrons/megatron-lm-boxin/lawrence/logs/query/n2000/${INDEX_STR//,/$''}__t${NTRAIN}"

mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

# IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/faissgpu"
IMAGE="nvcr.io/nvidia/pytorch:22.04-py3"
PARTITION="batch_short_dgx2h_m2"
# PARTITION="batch_dgx2h_m2"

# INDEX_STR_SIMPLE="${INDEX_STR//,/$''}"
# echo "INDEX_STR = $INDEX_STR"
# echo "INDEX_STR_SIMPLE = ${INDEX_STR_SIMPLE}"
# echo "LOG_DIR   = $LOG_DIR"
# exit

# for i in $(seq 0 1 15); do
# for i in {00..15}; do
for i in {00..00}; do

    start=$((10#$i*127328))

    # echo "i = $i, start = $start, index_str $INDEX_STR."
    # continue
    # exit 0

    # --logdir /home/lmcafee-src/megatrons/megatron-lm-boxin/lawrence/logs/ \
    # --duration 2
    submit_job \
        --outfile "$LOG_DIR/${i}.log" \
        --email_mode never \
        --image $IMAGE \
        --setenv LD_PRELOAD="/opt/conda/lib/libmkl_core.so:/opt/conda/lib/libmkl_sequential.so" \
        --duration 2 \
        --partition $PARTITION \
        --mounts "$SHARE_DATA,$SHARE_SOURCE,$SHARE_OUTPUT" \
        --gpu 16 \
        -c "pip install h5py; \
            conda install -c conda-forge -y faiss-gpu; \
            python faiss/query_index.py \
            --index-path $INDEX_PATH \
            --feature-path $DATA_DIR/v2/data1/feat/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.tokens.00${i}.feat.hdf5 \
            --doc-path $DATA_DIR/v2/data1/wiki.train.h5py_start_0_end_2037248_ns_2037248_sl2048_seed_1234_with_offset.doc_ids.pkl \
            --chunk-path $DATA_DIR/v2/data1/pretraining_corpus.chunks.hdf5 \
            --out-dir $OUT_DIR \
            --out-prefix ${i} \
            --target-k 2000 --k 10000 \
            --start 0 --split 10 --efsearch 32 --nprobe 4096 --offset ${start}"
done

# eof
