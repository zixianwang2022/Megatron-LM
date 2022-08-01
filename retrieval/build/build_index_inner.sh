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
    # profile_stage_stop="pqs"
    # [x] profile_stage_stop="[ignore]"

    profile_stage_stop="preprocess"
    # profile_stage_stop="cluster"

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
    # ntrain=2500000 nadd=20000000 ncluster=262144 hnsw=32
    # ntrain=2500000 nadd=100000000 ncluster=262144 hnsw=32
    ntrain=2500000 nadd=10000000 ncluster=262144 hnsw=32

    pq_dim=32
    ivf_dim=256

    # data_ty=wiki
    data_ty=corpus

    index_ty=faiss-mono
    # index_ty=faiss-decomp
    # index_str="OPQ32_256,IVF${ncluster}_HNSW${hnsw},PQ32"

    if [ "0" -eq "1" ]; then

	# --index-str ${index_str} \
        cmd="python -m lawrence.build.build_index \
                    --task ${task} \
                    --data-ty ${data_ty} \
                    --ntrain ${ntrain} \
                    --nadd ${nadd} \
                    --ncluster ${ncluster} \
                    --hnsw-m ${hnsw} \
                    --ivf-dim ${ivf_dim} \
                    --pq-m ${pq_dim} \
                    --index-ty ${index_ty} \
                    --profile-stage-stop ${profile_stage_stop} \
                    --profile-single-encoder 0 \
    		    "
	# cmd="ls -alh"

    else
	PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-index-add
	cmd="python -m torch.distributed.launch \
    		    --nproc_per_node 8 \
		    --nnodes 1 \
                    --node_rank 0 \
                    --master_addr localhost \
                    --master_port 6000 \
		    ${SHARE_SOURCE}/megatrons/megatron-lm-retrieval-index-add/retrieval/build/build_index.py \
                    --task ${task} \
                    --data-ty ${data_ty} \
                    --ntrain ${ntrain} \
                    --nadd ${nadd} \
                    --ncluster ${ncluster} \
                    --hnsw-m ${hnsw} \
                    --ivf-dim ${ivf_dim} \
                    --pq-m ${pq_dim} \
                    --index-ty ${index_ty} \
                    --profile-stage-stop ${profile_stage_stop} \
                    --profile-single-encoder 0 \
    		    "
    fi

fi

if [ "0" -eq "1" ]; then
    pip install h5py
    conda install -c conda-forge -y faiss-gpu
fi

unset NCCL_DEBUG
echo "CMD = $cmd"
eval $cmd
exit 0

# eof
