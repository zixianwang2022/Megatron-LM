#!/bin/bash

set -u

# MODEL_FAMILY=llama OMIT_ARGS_ALL="xxxxxxxxx"
MODEL_FAMILY=megatron OMIT_ARGS_ALL="xxxxxxxxx use-llama-rotary-emb use-llama-qkv use-llama-mlp use-llama-matmul use-llama-default-dtype"
MODEL_TYPE=text
# MODEL_SIZES=70b
MODEL_SIZES="7b 13b 70b"


SCRIPT_DIR="/lustre/fs3/portfolios/adlr/users/lmcafee/llama/2/src/megatron-lm-llama2-loader/scripts/bigbench"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p ${LOG_DIR}

STANDARD_TASKS="abstract_narrative_understanding general_knowledge human_organs_senses intent_recognition riddle_sense similarities_abstraction simple_arithmetic_json simple_arithmetic_json_multiple_choice undo_permutation unit_conversion qa_wikidata linguistic_mappings date_understanding conlang_translation"
# TYDIQA_TASKS="tydiqa_goldp.ar tydiqa_goldp.bn tydiqa_goldp.en tydiqa_goldp.fi tydiqa_goldp.id tydiqa_goldp.ko tydiqa_goldp.ru tydiqa_goldp.sw tydiqa_goldp.te"

# STANDARD_TASKS="general_knowledge"
# STANDARD_TASKS="abstract_narrative_understanding human_organs_senses"

# TASK_LISTS="STANDARD_TASKS TYDIQA_TASKS"
TASK_LISTS="STANDARD_TASKS"
declare -A MY_DICT
MY_DICT["STANDARD_TASKS"]=" --max_length=64 --json_shots='0,1,2' "
MY_DICT["TYDIQA_TASKS"]=" --max_length=16 --json_shots='1,4' "

# model_name_list="gpt3-43b-multi-1.1t-gtc" 
# for model_name in $model_name_list; do
for TASKS in $TASK_LISTS; do
    for TASK in ${!TASKS}; do
	for MODEL_SIZE in $MODEL_SIZES; do
	    for OMIT_ARGS in ${OMIT_ARGS_ALL}; do
		TAG="${TASKS}_${TASK}_${MODEL_FAMILY}-${MODEL_TYPE}-${MODEL_SIZE}_OMIT:${OMIT_ARGS}"
		# TAG="${TASK}_${MODEL_FAMILY}-${MODEL_TYPE}-${MODEL_SIZE}_OMIT:${OMIT_ARGS}"
		echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		# echo "TASKS : ${TASKS}"
		# echo "TASK  : ${TASK}"
		echo "... TAG       : ${TAG}"
		echo "... OMIT_ARGS : ${OMIT_ARGS}"
		echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

		if [ "${MODEL_SIZE}" = "7b" ]; then
		    NPROCS=1
		elif [ "${MODEL_SIZE}" = "13b" ]; then
		    NPROCS=2
		elif [ "${MODEL_SIZE}" = "70b" ]; then
		    NPROCS=8
		else
		    echo "specialize for model size '${MODEL_SIZE}'."
		    exit 1
		fi

		sbatch --export=TASK_ARGS=" --task=${TASK} ${MY_DICT[$TASKS]}",MODEL_FAMILY="${MODEL_FAMILY}",MODEL_TYPE="${MODEL_TYPE}",MODEL_SIZE="${MODEL_SIZE}",SCRIPT_DIR="${SCRIPT_DIR}",OMIT_ARGS="${OMIT_ARGS}" \
		--output="${LOG_DIR}/${TAG}__%j.log" \
		--job-name="adlr-nlp:llama-${TAG}" \
		--ntasks-per-node=${NPROCS} \
		${SCRIPT_DIR}/job.sh
	    done
        done
    done
done
# done

# eof
