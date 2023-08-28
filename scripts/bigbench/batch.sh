#!/bin/bash

set -u

# MODEL_FAMILY=llama OMIT_ARGS_ALL="xxxxxxxxx"
# MODEL_FAMILY=megatron OMIT_ARGS_ALL="xxxxxxxxx use-llama-rotary-emb use-llama-qkv use-llama-mlp use-llama-matmul use-llama-default-dtype"

MODEL_FAMILY=llama EXTRA_ARGS_ALL="--log-world-size-to-tensorboard"
# MODEL_FAMILY=hf EXTRA_ARGS_ALL="--log-world-size-to-tensorboard"
# MODEL_FAMILY=megatron EXTRA_ARGS_ALL="--log-world-size-to-tensorboard"
# MODEL_FAMILY=megatron EXTRA_ARGS_ALL="--log-world-size-to-tensorboard --use-llama-qkv --use-llama-mlp --use-llama-matmul --use-llama-default-dtype --use-llama-qkv,--use-llama-mlp,--use-llama-matmul,--use-llama-default-dtype"
MODEL_TYPE=text
# MODEL_SIZES=7b
# MODEL_SIZES="7b 13b"
MODEL_SIZES="7b 13b 70b"

SCRIPT_DIR="/lustre/fs3/portfolios/adlr/users/lmcafee/llama/2/src/megatron-lm-llama2-loader/scripts/bigbench"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p ${LOG_DIR}

STANDARD_TASKS="abstract_narrative_understanding general_knowledge human_organs_senses intent_recognition riddle_sense similarities_abstraction simple_arithmetic_json simple_arithmetic_json_multiple_choice undo_permutation unit_conversion qa_wikidata linguistic_mappings date_understanding conlang_translation"
TYDIQA_TASKS="tydiqa_goldp.ar tydiqa_goldp.bn tydiqa_goldp.en tydiqa_goldp.fi tydiqa_goldp.id tydiqa_goldp.ko tydiqa_goldp.ru tydiqa_goldp.sw tydiqa_goldp.te"
xcopa_task_list="xcopa.en-template-mGPT-remove-punctuation xcopa.et-template-mGPT-remove-punctuation xcopa.ht-template-mGPT-remove-punctuation xcopa.id-template-mGPT-remove-punctuation xcopa.it-template-mGPT-remove-punctuation xcopa.qu-template-mGPT-remove-punctuation xcopa.sw-template-mGPT-remove-punctuation xcopa.ta-template-mGPT-remove-punctuation xcopa.tr-template-mGPT-remove-punctuation xcopa.vi-template-mGPT-remove-punctuation xcopa.zh-template-mGPT-remove-punctuation xcopa.th-template-mGPT-remove-punctuation"
flores_task_list="flores101.src-en-tgt-es flores101.src-en-tgt-de flores101.src-en-tgt-fr flores101.src-en-tgt-pt flores101.src-en-tgt-it flores101.src-en-tgt-pl flores101.src-en-tgt-id flores101.src-en-tgt-el flores101.src-en-tgt-nl flores101.src-en-tgt-vi flores101.src-en-tgt-fa flores101.src-en-tgt-tr flores101.src-en-tgt-cs flores101.src-en-tgt-ar flores101.src-en-tgt-sv flores101.src-en-tgt-ro flores101.src-en-tgt-uk flores101.src-en-tgt-hu flores101.src-en-tgt-fi flores101.src-es-tgt-en flores101.src-es-tgt-de flores101.src-es-tgt-fr flores101.src-es-tgt-pt flores101.src-es-tgt-it flores101.src-es-tgt-pl flores101.src-es-tgt-id flores101.src-es-tgt-el flores101.src-es-tgt-nl flores101.src-es-tgt-vi flores101.src-es-tgt-fa flores101.src-es-tgt-tr flores101.src-es-tgt-cs flores101.src-es-tgt-ar flores101.src-es-tgt-sv flores101.src-es-tgt-ro flores101.src-es-tgt-uk flores101.src-es-tgt-hu flores101.src-es-tgt-fi flores101.src-de-tgt-en flores101.src-de-tgt-es flores101.src-de-tgt-fr flores101.src-de-tgt-pt flores101.src-de-tgt-it flores101.src-de-tgt-pl flores101.src-de-tgt-id flores101.src-de-tgt-el flores101.src-de-tgt-nl flores101.src-de-tgt-vi flores101.src-de-tgt-fa flores101.src-de-tgt-tr flores101.src-de-tgt-cs flores101.src-de-tgt-ar flores101.src-de-tgt-sv flores101.src-de-tgt-ro flores101.src-de-tgt-uk flores101.src-de-tgt-hu flores101.src-de-tgt-fi flores101.src-fr-tgt-en flores101.src-fr-tgt-es flores101.src-fr-tgt-de flores101.src-fr-tgt-pt flores101.src-fr-tgt-it flores101.src-fr-tgt-pl flores101.src-fr-tgt-id flores101.src-fr-tgt-el flores101.src-fr-tgt-nl flores101.src-fr-tgt-vi flores101.src-fr-tgt-fa flores101.src-fr-tgt-tr flores101.src-fr-tgt-cs flores101.src-fr-tgt-ar flores101.src-fr-tgt-sv flores101.src-fr-tgt-ro flores101.src-fr-tgt-uk flores101.src-fr-tgt-hu flores101.src-fr-tgt-fi flores101.src-pt-tgt-en flores101.src-pt-tgt-es flores101.src-pt-tgt-de flores101.src-pt-tgt-fr flores101.src-pt-tgt-it flores101.src-pt-tgt-pl flores101.src-pt-tgt-id flores101.src-pt-tgt-el flores101.src-pt-tgt-nl flores101.src-pt-tgt-vi flores101.src-pt-tgt-fa flores101.src-pt-tgt-tr flores101.src-pt-tgt-cs flores101.src-pt-tgt-ar flores101.src-pt-tgt-sv flores101.src-pt-tgt-ro flores101.src-pt-tgt-uk flores101.src-pt-tgt-hu flores101.src-pt-tgt-fi flores101.src-it-tgt-en flores101.src-it-tgt-es flores101.src-it-tgt-de flores101.src-it-tgt-fr flores101.src-it-tgt-pt flores101.src-it-tgt-pl flores101.src-it-tgt-id flores101.src-it-tgt-el flores101.src-it-tgt-nl flores101.src-it-tgt-vi flores101.src-it-tgt-fa flores101.src-it-tgt-tr flores101.src-it-tgt-cs flores101.src-it-tgt-ar flores101.src-it-tgt-sv flores101.src-it-tgt-ro flores101.src-it-tgt-uk flores101.src-it-tgt-hu flores101.src-it-tgt-fi flores101.src-pl-tgt-en flores101.src-pl-tgt-es flores101.src-pl-tgt-de flores101.src-pl-tgt-fr flores101.src-pl-tgt-pt flores101.src-pl-tgt-it flores101.src-pl-tgt-id flores101.src-pl-tgt-el flores101.src-pl-tgt-nl flores101.src-pl-tgt-vi flores101.src-pl-tgt-fa flores101.src-pl-tgt-tr flores101.src-pl-tgt-cs flores101.src-pl-tgt-ar flores101.src-pl-tgt-sv flores101.src-pl-tgt-ro flores101.src-pl-tgt-uk flores101.src-pl-tgt-hu flores101.src-pl-tgt-fi flores101.src-id-tgt-en flores101.src-id-tgt-es flores101.src-id-tgt-de flores101.src-id-tgt-fr flores101.src-id-tgt-pt flores101.src-id-tgt-it flores101.src-id-tgt-pl flores101.src-id-tgt-el flores101.src-id-tgt-nl flores101.src-id-tgt-vi flores101.src-id-tgt-fa flores101.src-id-tgt-tr flores101.src-id-tgt-cs flores101.src-id-tgt-ar flores101.src-id-tgt-sv flores101.src-id-tgt-ro flores101.src-id-tgt-uk flores101.src-id-tgt-hu flores101.src-id-tgt-fi flores101.src-el-tgt-en flores101.src-el-tgt-es flores101.src-el-tgt-de flores101.src-el-tgt-fr flores101.src-el-tgt-pt flores101.src-el-tgt-it flores101.src-el-tgt-pl flores101.src-el-tgt-id flores101.src-el-tgt-nl flores101.src-el-tgt-vi flores101.src-el-tgt-fa flores101.src-el-tgt-tr flores101.src-el-tgt-cs flores101.src-el-tgt-ar flores101.src-el-tgt-sv flores101.src-el-tgt-ro flores101.src-el-tgt-uk flores101.src-el-tgt-hu flores101.src-el-tgt-fi flores101.src-nl-tgt-en flores101.src-nl-tgt-es flores101.src-nl-tgt-de flores101.src-nl-tgt-fr flores101.src-nl-tgt-pt flores101.src-nl-tgt-it flores101.src-nl-tgt-pl flores101.src-nl-tgt-id flores101.src-nl-tgt-el flores101.src-nl-tgt-vi flores101.src-nl-tgt-fa flores101.src-nl-tgt-tr flores101.src-nl-tgt-cs flores101.src-nl-tgt-ar flores101.src-nl-tgt-sv flores101.src-nl-tgt-ro flores101.src-nl-tgt-uk flores101.src-nl-tgt-hu flores101.src-nl-tgt-fi flores101.src-vi-tgt-en flores101.src-vi-tgt-es flores101.src-vi-tgt-de flores101.src-vi-tgt-fr flores101.src-vi-tgt-pt flores101.src-vi-tgt-it flores101.src-vi-tgt-pl flores101.src-vi-tgt-id flores101.src-vi-tgt-el flores101.src-vi-tgt-nl flores101.src-vi-tgt-fa flores101.src-vi-tgt-tr flores101.src-vi-tgt-cs flores101.src-vi-tgt-ar flores101.src-vi-tgt-sv flores101.src-vi-tgt-ro flores101.src-vi-tgt-uk flores101.src-vi-tgt-hu flores101.src-vi-tgt-fi flores101.src-fa-tgt-en flores101.src-fa-tgt-es flores101.src-fa-tgt-de flores101.src-fa-tgt-fr flores101.src-fa-tgt-pt flores101.src-fa-tgt-it flores101.src-fa-tgt-pl flores101.src-fa-tgt-id flores101.src-fa-tgt-el flores101.src-fa-tgt-nl flores101.src-fa-tgt-vi flores101.src-fa-tgt-tr flores101.src-fa-tgt-cs flores101.src-fa-tgt-ar flores101.src-fa-tgt-sv flores101.src-fa-tgt-ro flores101.src-fa-tgt-uk flores101.src-fa-tgt-hu flores101.src-fa-tgt-fi flores101.src-tr-tgt-en flores101.src-tr-tgt-es flores101.src-tr-tgt-de flores101.src-tr-tgt-fr flores101.src-tr-tgt-pt flores101.src-tr-tgt-it flores101.src-tr-tgt-pl flores101.src-tr-tgt-id flores101.src-tr-tgt-el flores101.src-tr-tgt-nl flores101.src-tr-tgt-vi flores101.src-tr-tgt-fa flores101.src-tr-tgt-cs flores101.src-tr-tgt-ar flores101.src-tr-tgt-sv flores101.src-tr-tgt-ro flores101.src-tr-tgt-uk flores101.src-tr-tgt-hu flores101.src-tr-tgt-fi flores101.src-cs-tgt-en flores101.src-cs-tgt-es flores101.src-cs-tgt-de flores101.src-cs-tgt-fr flores101.src-cs-tgt-pt flores101.src-cs-tgt-it flores101.src-cs-tgt-pl flores101.src-cs-tgt-id flores101.src-cs-tgt-el flores101.src-cs-tgt-nl flores101.src-cs-tgt-vi flores101.src-cs-tgt-fa flores101.src-cs-tgt-tr flores101.src-cs-tgt-ar flores101.src-cs-tgt-sv flores101.src-cs-tgt-ro flores101.src-cs-tgt-uk flores101.src-cs-tgt-hu flores101.src-cs-tgt-fi flores101.src-ar-tgt-en flores101.src-ar-tgt-es flores101.src-ar-tgt-de flores101.src-ar-tgt-fr flores101.src-ar-tgt-pt flores101.src-ar-tgt-it flores101.src-ar-tgt-pl flores101.src-ar-tgt-id flores101.src-ar-tgt-el flores101.src-ar-tgt-nl flores101.src-ar-tgt-vi flores101.src-ar-tgt-fa flores101.src-ar-tgt-tr flores101.src-ar-tgt-cs flores101.src-ar-tgt-sv flores101.src-ar-tgt-ro flores101.src-ar-tgt-uk flores101.src-ar-tgt-hu flores101.src-ar-tgt-fi flores101.src-sv-tgt-en flores101.src-sv-tgt-es flores101.src-sv-tgt-de flores101.src-sv-tgt-fr flores101.src-sv-tgt-pt flores101.src-sv-tgt-it flores101.src-sv-tgt-pl flores101.src-sv-tgt-id flores101.src-sv-tgt-el flores101.src-sv-tgt-nl flores101.src-sv-tgt-vi flores101.src-sv-tgt-fa flores101.src-sv-tgt-tr flores101.src-sv-tgt-cs flores101.src-sv-tgt-ar flores101.src-sv-tgt-ro flores101.src-sv-tgt-uk flores101.src-sv-tgt-hu flores101.src-sv-tgt-fi flores101.src-ro-tgt-en flores101.src-ro-tgt-es flores101.src-ro-tgt-de flores101.src-ro-tgt-fr flores101.src-ro-tgt-pt flores101.src-ro-tgt-it flores101.src-ro-tgt-pl flores101.src-ro-tgt-id flores101.src-ro-tgt-el flores101.src-ro-tgt-nl flores101.src-ro-tgt-vi flores101.src-ro-tgt-fa flores101.src-ro-tgt-tr flores101.src-ro-tgt-cs flores101.src-ro-tgt-ar flores101.src-ro-tgt-sv flores101.src-ro-tgt-uk flores101.src-ro-tgt-hu flores101.src-ro-tgt-fi flores101.src-uk-tgt-en flores101.src-uk-tgt-es flores101.src-uk-tgt-de flores101.src-uk-tgt-fr flores101.src-uk-tgt-pt flores101.src-uk-tgt-it flores101.src-uk-tgt-pl flores101.src-uk-tgt-id flores101.src-uk-tgt-el flores101.src-uk-tgt-nl flores101.src-uk-tgt-vi flores101.src-uk-tgt-fa flores101.src-uk-tgt-tr flores101.src-uk-tgt-cs flores101.src-uk-tgt-ar flores101.src-uk-tgt-sv flores101.src-uk-tgt-ro flores101.src-uk-tgt-hu flores101.src-uk-tgt-fi flores101.src-hu-tgt-en flores101.src-hu-tgt-es flores101.src-hu-tgt-de flores101.src-hu-tgt-fr flores101.src-hu-tgt-pt flores101.src-hu-tgt-it flores101.src-hu-tgt-pl flores101.src-hu-tgt-id flores101.src-hu-tgt-el flores101.src-hu-tgt-nl flores101.src-hu-tgt-vi flores101.src-hu-tgt-fa flores101.src-hu-tgt-tr flores101.src-hu-tgt-cs flores101.src-hu-tgt-ar flores101.src-hu-tgt-sv flores101.src-hu-tgt-ro flores101.src-hu-tgt-uk flores101.src-hu-tgt-fi flores101.src-fi-tgt-en flores101.src-fi-tgt-es flores101.src-fi-tgt-de flores101.src-fi-tgt-fr flores101.src-fi-tgt-pt flores101.src-fi-tgt-it flores101.src-fi-tgt-pl flores101.src-fi-tgt-id flores101.src-fi-tgt-el flores101.src-fi-tgt-nl flores101.src-fi-tgt-vi flores101.src-fi-tgt-fa flores101.src-fi-tgt-tr flores101.src-fi-tgt-cs flores101.src-fi-tgt-ar flores101.src-fi-tgt-sv flores101.src-fi-tgt-ro flores101.src-fi-tgt-uk flores101.src-fi-tgt-hu"
tydiqa_task_list="tydiqa_goldp.ar tydiqa_goldp.bn tydiqa_goldp.en tydiqa_goldp.fi tydiqa_goldp.id tydiqa_goldp.ko tydiqa_goldp.ru tydiqa_goldp.sw tydiqa_goldp.te"
mgsm_task_list="mgsm.en-english_cot mgsm.de-english_cot mgsm.fr-english_cot mgsm.es-english_cot mgsm.ru-english_cot mgsm.zh-english_cot mgsm.ja-english_cot mgsm.th-english_cot mgsm.te-english_cot mgsm.bn-english_cot mgsm.sw-english_cot"


# STANDARD_TASKS="general_knowledge"
# xcopa_task_list="xcopa.en-template-mGPT-remove-punctuation"

TASK_LISTS="STANDARD_TASKS"
# TASK_LISTS="TYDIQA_TASKS"
# TASK_LISTS="xcopa_task_list"
# TASK_LISTS="tydiqa_task_list"
# TASK_LISTS="flores_task_list"
# TASK_LISTS="mgsm_task_list"


declare -A MY_DICT
MY_DICT["STANDARD_TASKS"]=" --max_length=64 --json_shots='0,1,2' "
MY_DICT["TYDIQA_TASKS"]=" --max_length=16 --json_shots='1,4' "
MY_DICT["flores_task_list"]=" --max_length=64 --json_shots='1,4,32' "
MY_DICT["xcopa_task_list"]=" --max_length=64 --json_shots='0,4,32' "
MY_DICT["tydiqa_task_list"]=" --max_length=16 --json_shots='1,4' "
MY_DICT["mgsm_task_list"]=" --max_length=256 --json_shots=0 "

# model_name_list="gpt3-43b-multi-1.1t-gtc" 
# for model_name in $model_name_list; do
for TASKS in $TASK_LISTS; do
    for TASK in ${!TASKS}; do
	for MODEL_SIZE in $MODEL_SIZES; do
	    # for OMIT_ARGS in ${OMIT_ARGS_ALL}; do
	    for EXTRA_ARGS in ${EXTRA_ARGS_ALL}; do
		# TAG="${TASKS}_${TASK}_${MODEL_FAMILY}-${MODEL_TYPE}-${MODEL_SIZE}_OMIT:${OMIT_ARGS}"
		TAG="${TASKS}_${TASK}_${MODEL_FAMILY}-${MODEL_TYPE}-${MODEL_SIZE}_EXTRA:${EXTRA_ARGS}"
		# TAG="${TASK}_${MODEL_FAMILY}-${MODEL_TYPE}-${MODEL_SIZE}_OMIT:${OMIT_ARGS}"
		echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		# echo "TASKS : ${TASKS}"
		# echo "TASK  : ${TASK}"
		echo "... TAG       : ${TAG}"
		# echo "... OMIT_ARGS : ${OMIT_ARGS}"
		echo "... EXTRA_ARGS : ${EXTRA_ARGS}"
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

		# sbatch --export=TASK_ARGS=" --task=${TASK} ${MY_DICT[$TASKS]}",MODEL_FAMILY="${MODEL_FAMILY}",MODEL_TYPE="${MODEL_TYPE}",MODEL_SIZE="${MODEL_SIZE}",SCRIPT_DIR="${SCRIPT_DIR}",OMIT_ARGS="${OMIT_ARGS}" \
		sbatch --export=TASK_ARGS=" --task=${TASK} ${MY_DICT[$TASKS]}",MODEL_FAMILY="${MODEL_FAMILY}",MODEL_TYPE="${MODEL_TYPE}",MODEL_SIZE="${MODEL_SIZE}",SCRIPT_DIR="${SCRIPT_DIR}",EXTRA_ARGS="${EXTRA_ARGS}" \
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
