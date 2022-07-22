#!/bin/bash

# ~~~~~~~~ debug ~~~~~~~~
# ngpus=$(nvidia-smi --list-gpus | wc -l)
# echo "debug: SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE."
# echo "debug: SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK."
# echo "debug: SLURM_JOB_ID = $SLURM_JOB_ID."
# echo "debug: SLURM_JOB_NAME = $SLURM_JOB_NAME."
# echo "debug: SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST."
# echo "debug: SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES."
# echo "debug: SLURM_LOCALID = $SLURM_LOCALID."
# echo "debug: SLURM_NODEID = $SLURM_NODEID."
# echo "debug: SLURM_NPROCS = $SLURM_NPROCS."
# echo "debug: SLURM_PROCID = $SLURM_PROCID."
# echo "debug: ngpus = $ngpus."
# exit 0

# ~~~~~~~~ batch mode ~~~~~~~~
if [ "0" -eq "1" ]; then

    if [ "$#" -ne 1 ]; then
        echo "illegal num args, $#."
        exit 1
    fi
    cmd=$1

# ~~~~~~~~ interactive mode ~~~~~~~~
else

    ncenters=1e2 # *1e4
    ntrain=1e4 # *1e6
    cmd="python -m lawrence.sandbox.test_cluster_multinode \
                --ncenters ${ncenters} \
                --ntrain ${ntrain} \
                "

fi

# ~~~~~~~~ packages ~~~~~~~~
if [ "0" -eq "1" ]; then
    # apt install lsof
    # pip install mpi4py
    pip install asyncssh
    pip install h5py
    conda install -c conda-forge -y faiss-gpu
    # conda install -c conda-forge cudatoolkit=11.5
    # conda install -c rapidsai -c nvidia -c conda-forge dask-cuda cudatoolkit=11.5
    # pip install dask-cuda
    # conda install dask dask-core dask-cuda
fi

# ~~~~~~~~ eval ~~~~~~~~
# unset NCCL_DEBUG
NCCL_DEBUG=WARN
# eval $cmd
# su
# kill $(lsof -t -i:6789) # kill -9
# eval "$cmd --role server &"
if [ "$SLURM_NODEID" -eq "0" ]; then
    eval "$cmd --role client" # &"
fi

# eof
