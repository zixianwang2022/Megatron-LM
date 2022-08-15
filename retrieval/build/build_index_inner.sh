#!/bin/bash

set -u

DIR=$(readlink -f `pwd`)
# source $SHARE_SOURCE/megatrons/megatron-lm-retrieval-index-add/retrieval/build/build_index_cmd.sh
source $DIR/retrieval/build/build_index_cmd.sh

if [ "1" -eq "1" ]; then
    pip install h5py
    conda install -c conda-forge -y faiss-gpu
fi

unset NCCL_DEBUG
echo "DIR = '$DIR'."
echo "BUILD_INDEX_CMD = '$BUILD_INDEX_CMD'."
eval $BUILD_INDEX_CMD
exit 0

# eof
