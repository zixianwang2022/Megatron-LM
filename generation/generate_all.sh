CUDA_VISIBLE_DEVICES=0 MASTER_PORT=44140 source generation/generate_flamingo_2b.sh 0 &
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=44141 source generation/generate_flamingo_2b.sh 1 &
CUDA_VISIBLE_DEVICES=2 MASTER_PORT=44142 source generation/generate_flamingo_2b.sh 2 &
CUDA_VISIBLE_DEVICES=3 MASTER_PORT=44143 source generation/generate_flamingo_2b.sh 3 &
CUDA_VISIBLE_DEVICES=4 MASTER_PORT=44144 source generation/generate_flamingo_2b.sh 4 &
CUDA_VISIBLE_DEVICES=5 MASTER_PORT=44145 source generation/generate_flamingo_2b.sh 5 &
CUDA_VISIBLE_DEVICES=6 MASTER_PORT=44146 source generation/generate_flamingo_2b.sh 6 &
CUDA_VISIBLE_DEVICES=7 MASTER_PORT=44147 source generation/generate_flamingo_2b.sh 7 



# srun -p interactive,batch_block1 -A llmservice_nlp_fm -t 2:00:00 --job-name=adlr-nlp-largelm:test-4 --container-mounts="/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1,/lustre/fs2:/lustre/fs2,/lustre/fs3:/lustre/fs3,/lustre/fs4:/lustre/fs4,/home/guilinl:/home/guilinl" --container-image="/home/guilinl/sd_video/general/image/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels_pyspy.sqsh" --gres=gpu:8 --ntasks=1 --cpus-per-task=64 --exclusive --nodes=1 --mem=0  --pty bash