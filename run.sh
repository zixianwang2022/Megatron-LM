# interactive shell
#
#srun -p luna,interactive -A adlr -t 0:30:00 --job-name=adlr-nlp-largelm:cpu --container-mounts=/lustre/fsw/adlr:/lustre/fsw/adlr --container-image="/lustre/fsw/adlr/adlr-nlp/boxinw/images/retro.sqsh"   --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  --pty bash
#srun -p luna,interactive -A adlr -t 0:30:00 --job-name=adlr-nlp-largelm:cpu --container-mounts=/lustre/fsw/adlr:/lustre/fsw/adlr --container-image="/lustre/fsw/adlr/adlr-nlp/boxinw/images/retrov2.sqsh" --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  --pty bash
#
#
#sbatch scripts/pretrain-nextlm-800m-gpt.sh
#sbatch scripts/pretrain-nextlm-800m-gpt-lr-2e-6.sh
#sbatch scripts/pretrain-nextlm-800m-retro.sh
#
#sbatch scripts/pretrain-nextlm-8b-retro.sh
#
## eval ppl
#
#/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm

sbatch scripts/evalppl-nextlm-800m-gpt.sh gpt3-800m-pretraining-gpt-fitting
sbatch scripts/evalppl-nextlm-800m-gpt.sh gpt3-843m-multi-1.1t-gtc-llr
sbatch scripts/evalppl-nextlm-800m-retro.sh gpt3-800m-pretraining-retro-fitting

#sbatch scripts/evalppl-nextlm-2b-gpt.sh gpt3-2b-multi-1.1t-gtc
#sbatch scripts/evalppl-nextlm-2b-gpt.sh gpt3-2b-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-2b-retro.sh gpt3-2b-pretraining-retro-fitting
#
#sbatch scripts/evalppl-nextlm-8b-gpt.sh gpt3-8b-multi-1.1t-gtc
#sbatch scripts/evalppl-nextlm-8b-gpt.sh gpt3-8b-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-8b-retro.sh gpt3-8b-pretraining-retro-fitting-noseqpar
#
#sbatch scripts/evalppl-nextlm-22b-gpt.sh gpt3-22b-multi-1.1t-gtc
#sbatch scripts/evalppl-nextlm-22b-gpt.sh gpt3-22b-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-22b-retro.sh gpt3-22b-pretraining-retro-fitting-noseqpar
#
#sbatch scripts/evalppl-nextlm-43b-gpt.sh gpt3-43b-multi-1.1t-gtc-tp8pp4vp1
#sbatch scripts/evalppl-nextlm-43b-gpt.sh gpt3-43b-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-43b-retro.sh gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed

#/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm
#
#sbatch scripts/evalppl-nextlm-800m-retro.sh gpt3-800m-pretraining-retro-fitting
#
#
#sbatch scripts/evalppl-nextlm-8b-retro.sh gpt3-8b-pretraining-retro-fitting
#
#python tools/checkpoint_util.py \
#        --model-type GPT \
#        --load-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting \
#        --save-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1 \
#        --target-tensor-parallel-size 8 \
#        --target-pipeline-parallel-size 1
