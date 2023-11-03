#! /bin/bash

DATA_PATH=/lustre/fsw/adlr/adlr-nlp/adlr_ci/megatron/data/captioning_ft.yaml
CHECKPOINT_PATH=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingo_2b_captioning_SAM_ft/checkpoints
TENSORBOARD_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingo_2b_captioning_SAM_ft/tensorboard_logs
SCRIPTS_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/megatron-lm-flamingo-tests

USE_CORE=0
USE_TE=0
MBS=1
GBS=8
MAX_STEPS=50
NUM_NODES=1
TP_SIZE=1
PP_SIZE=1

bash ./tests/functional_tests/test_scripts/flamingo/finetune_flamingo_2b_captioning_SAM_test.sh DATA_PATH=$DATA_PATH CHECKPOINT_PATH=$CHECKPOINT_PATH TENSORBOARD_DIR=$TENSORBOARD_DIR SCRIPTS_DIR=$SCRIPTS_DIR USE_TE=$USE_TE TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE VP_SIZE=$VP_SIZE NUM_NODES=$NUM_NODES MAX_STEPS=$MAX_STEPS USE_CORE=$USE_CORE MBS=$MBS GBS=$GBS

python tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py /lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingo_2b_captioning_SAM_ft/tensorboard_logs flamingo_2b_captioning_SAM_ft

export EXPECTED_METRICS_FILE=/lustre/fsw/adlr/adlr-nlp/adlr_ci/megatron/data/captioning_ft_results.json

export LOGS_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingo_2b_captioning_SAM_ft/tensorboard_logs

pytest tests/functional_tests/python_test_utils/test_ci_pipeline.py