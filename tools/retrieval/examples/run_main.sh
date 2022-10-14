#!/bin/bash

set -u

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
source $PARENT_PATH/get_cmd.sh

# >>>
# ... now required in image
# if [ "0" -eq "1" ]; then
#     pip install h5py
#     conda install -c conda-forge -y faiss-gpu
# fi
# <<<

unset NCCL_DEBUG
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "PARENT_PATH = '$PARENT_PATH'."
echo "RETRO_PREPROCESS_CMD = '$RETRO_PREPROCESS_CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $RETRO_PREPROCESS_CMD
