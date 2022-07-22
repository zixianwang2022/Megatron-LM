#!/bin/bash

set -u

# run-cluster-dask-jobs.sh [ entry ]
# run-dask-process.sh [ dask-scheduler, dask-cuda-worker ]
# wait_for_workers.py

# DIR=$SHARE_SOURCE/src/megatrons/megatron-lm-boxin/lawrence/sandbox/rapids
DIR=$SHARE_SOURCE/megatrons/megatron-lm-boxin/lawrence/sandbox/rapids
SCHEDULER_PATH=$DIR/scheduler.json
NUM_WORKERS=1

if [[ $SLURM_NODEID == 0 ]]; then

    echo "scheduler + worker."

    # dask-scheduler --scheduler-file=$SCHEDULER_PATH &
    # dask-cuda-worker --scheduler-file=$SCHEDULER_PATH &
    # python $DIR/wait_for_workers.py \
    #        --num-expected-workers $NUM_WORKERS \
    #        --scheduler-file-path $SCHEDULER_PATH
    python $DIR/test_kmeans.py

else

    echo "worker."
    exit 0

fi

# eof
