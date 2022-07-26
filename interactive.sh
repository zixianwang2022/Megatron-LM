srun -N 1 \
 --ntasks-per-node=8 \
 --time=60 \
 -p interactive \
 -A gpu-comparch \
 --container-image "/lustre/fsw/adlr/adlr-nlp/images/pytorch+bf16_nccl_fusion.sqsh" \
 --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr,/lustre/fsw/gpu-comparch:/lustre/fsw/gpu-comparch" \
 --job-name gpu-comparch-psx:interactive \
 --pty bash
