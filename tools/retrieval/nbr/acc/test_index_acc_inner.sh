#!/bin/bash

# DIR=$(readlink -f `pwd`)
# echo $DIR
# exit

# if [ "$#" -ne 3 ]; then
if [ "$#" -ne 2 ]; then
    echo "illegal num args, $#."
    exit 1
fi

task=$1
# index_path_0=$2
# index_path_1=$3
index_path=$2

pip install h5py
# conda install -c conda-forge -y faiss-gpu # ... temporary, test-knn only
# python -m lawrence.acc.test_index_acc --task $task --i0 $index_path_0 --i1 $index_path_1
python -m lawrence.acc.test_index_acc --task $task --index $index_path

# eof
