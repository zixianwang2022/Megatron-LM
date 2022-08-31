srun -N 1 \
 --ntasks-per-node=8 \
 --time=120 \
 -p interactive \
 -A gpu-comparch \
 --container-image "/lustre/fsw/adlr/adlr-nlp/images/pytorch+bf16_nccl_fusion.sqsh" \
 --container-mounts "/lustre/fsw/gpu-comparch:/lustre/fsw/gpu-comparch,/lustre/fsw/adlr/adlr-nlp:/lustre/fsw/adlr/adlr-nlp,/lustre/fsw/adlr/adlr-nlp-large:/lustre/fsw/adlr/adlr-nlp-large" \
 --job-name gpu-comparch-psx:interactive \
 --pty bash
