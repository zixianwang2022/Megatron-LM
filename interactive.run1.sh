srun \
	-A adlr \
	-p luna \
	-N 1 \
	--ntasks-per-node 8 \
	-J adlr-nlp:develop:interative.run1 \
	--container-image "/lustre/fsw/adlr/adlr-nlp/images/pytorch+bf16_nccl_fusion+pyspy.sqsh" \
	--container-mounts "/lustre/fsw/adlr/:/lustre/fsw/adlr/" \
	--pty bash


