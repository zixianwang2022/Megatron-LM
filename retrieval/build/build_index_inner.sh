#!/bin/bash

# DIR=$(readlink -f `pwd`)
# echo $DIR
# exit

if [ "0" -eq "1" ]; then
   if [ "$#" -ne 1 ]; then
       echo "illegal num args, $#."
       exit 1
   fi
   cmd=$1

else
    # >>>>>>>>>>>>>>>>>>>>>>>
    # [x] profile_stage_stop="data"
    # profile_stage_stop="opq"
    # profile_stage_stop="ivf"
    profile_stage_stop="pqs"
    # profile_stage_stop="[ignore]"

    # task="clean-data"
    # task="split-data"
    # task=train
    task=add

    # ntrain=2048 ncluster=64 hnsw=4
    # ntrain=131072 ncluster=128 hnsw=32
    # ntrain=5000000 ncluster=100000 hnsw=32
    # ntrain=15000000 ncluster=500000 hnsw=32
    # ntrain=20000000 ncluster=4194304 hnsw=32
    # ntrain=50000000 nadd=200000000 ncluster=4194304 hnsw=32
    # ntrain=300000000 ncluster=4194304 hnsw=32
    # ntrain=50000 nadd=20000000 ncluster=16384 hnsw=32
    ntrain=2500000 nadd=20000000 ncluster=262144 hnsw=32

    pq_dim=32
    ivf_dim=256

    # data_ty=wiki
    data_ty=corpus

    # index_ty=faiss-mono
    index_ty=faiss-decomp
    # index_str="OPQ32_256,IVF${ncluster}_HNSW${hnsw},PQ32"

    # --index-str ${index_str} \
    cmd="python -m lawrence.build.build_index \
                --task ${task} \
                --data-ty ${data_ty} \
                --ntrain ${ntrain} \
                --nadd ${nadd} \
                --ncluster ${ncluster} \
                --hnsw-dim ${hnsw} \
                --ivf-dim ${ivf_dim} \
                --pq-dim ${pq_dim} \
                --index-ty ${index_ty} \
                --profile-stage-stop ${profile_stage_stop} \
                --profile-single-encoder 0 \
    "
    # cmd="ls -alh"
    # <<<<<<<<<<<<<<<<<<<<<<<

fi

if [ "0" -eq "1" ]; then
    pip install h5py
    conda install -c conda-forge -y faiss-gpu
fi

echo "CMD = $cmd"
eval $cmd
exit 0

# eof
