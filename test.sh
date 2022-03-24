#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
	                  --nnodes 1 \
			                    --node_rank 0 \
					                      --master_addr localhost \
							      --master_port $((6000+$1)) \
										                  "

echo $CUDA_VISIBLE_DEVICES
echo $DISTRIBUTED_ARGS

INPUT_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/test_$2.json
echo $INPUT_PATH


