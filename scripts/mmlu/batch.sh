#!/bin/bash

set -u

# MODEL_SIZES="7b 13b 70b" MODEL_FAMILIES="llama megatron"
MODEL_SIZES="13b 70b" MODEL_FAMILIES="llama megatron"
MODEL_TYPE=text

SCRIPT_DIR="/lustre/fs3/portfolios/adlr/users/lmcafee/llama/2/src/megatron-lm-llama2-loader/scripts/mmlu"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p ${LOG_DIR}

for MODEL_SIZE in $MODEL_SIZES; do
    for MODEL_FAMILY in $MODEL_FAMILIES; do
	TAG="${MODEL_FAMILY}-${MODEL_TYPE}-${MODEL_SIZE}"
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	echo "... TAG       : ${TAG}"
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

	sbatch \
	    --export=MODEL_FAMILY="${MODEL_FAMILY}",MODEL_TYPE="${MODEL_TYPE}",MODEL_SIZE="${MODEL_SIZE}",SCRIPT_DIR="${SCRIPT_DIR}" \
	    --output="${LOG_DIR}/${TAG}__%j.log" \
	    --job-name="adlr-nlp-mmlu:${TAG}" \
	    --ntasks-per-node=${NPROCS} \
	    ${SCRIPT_DIR}/job.sh
    done
done

# eof
