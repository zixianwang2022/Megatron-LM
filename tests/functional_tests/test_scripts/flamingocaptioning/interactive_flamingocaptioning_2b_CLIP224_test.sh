#! /bin/bash

# These paths should be modified by user to use their source code and output dir
CHECKPOINT_PATH=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingocaptioning_2b_CLIP224_test/checkpoints
TENSORBOARD_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingocaptioning_2b_CLIP224_test/tensorboard_logs
SCRIPTS_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/megatron-lm-flamingo-tests

DATA_PATH=/lustre/fsw/adlr/adlr-nlp/adlr_ci/megatron/data/flamingo_data
mkdir /workspace/data
ln -s $DATA_PATH /workspace/data/flamingo_data

USE_CORE=0
USE_TE=0
MBS=4
GBS=32
MAX_STEPS=50
NUM_NODES=1
TP_SIZE=1
PP_SIZE=1
export ADDITIONAL_PARAMS="--load /lustre/fsw/adlr/adlr-nlp/adlr_ci/megatron/data/flamingo_data/gpt3-2b-multi-1.1t-gtc --visual-arch L_14 --visual-type vit --visual-path /lustre/fsw/adlr/adlr-nlp/adlr_ci/megatron/data/flamingo_data/vit_L_14 --img-h 224 --img-w 224"

bash ./tests/functional_tests/test_scripts/flamingocaptioning/finetune_flamingocaptioning_distributed_test.sh DATA_PATH=$DATA_PATH CHECKPOINT_PATH=$CHECKPOINT_PATH TENSORBOARD_DIR=$TENSORBOARD_DIR SCRIPTS_DIR=$SCRIPTS_DIR USE_TE=$USE_TE TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE VP_SIZE=$VP_SIZE NUM_NODES=$NUM_NODES MAX_STEPS=$MAX_STEPS USE_CORE=$USE_CORE MBS=$MBS GBS=$GBS

python tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py /lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingocaptioning_2b_CLIP224_test/tensorboard_logs flamingocaptioning_2b_CLIP224_test

export EXPECTED_METRICS_FILE=tests/functional_tests/test_results/flamingocaptioning/flamingocaptioning_2b_CLIP224_test.json

export LOGS_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingocaptioning_2b_CLIP224_test/tensorboard_logs

pytest tests/functional_tests/python_test_utils/test_ci_pipeline.py