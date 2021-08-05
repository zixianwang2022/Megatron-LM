#!/bin/bash

#SBATCH -p luna -A adlr -t 0:20:00 --nodes=35 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=adlr-nlp-develop:gpt3-530b

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

DATAPATH="/lustre/fsw/adlr/adlr-nlp/data/gpt2_indexed_dataset"

options=" \
	  --tensor-model-parallel-size 8 \
	  --pipeline-model-parallel-size 35 \
	  --num-layers 105 \
	  --hidden-size 20480 \
	  --num-attention-heads 128 \
	  --seq-length 2048 \
	  --max-position-embeddings 2048 \
	  --micro-batch-size 1 \
	  --global-batch-size 192 \
	  --train-iters 500 \
	  --lr-decay-iters 320 \
	  --lr 0.000015 \
	  --min-lr 0.00001 \
	  --lr-decay-style cosine \
	  --lr-warmup-fraction 0.01 \
	  --data-path ${DATAPATH}/roberta_dataset/rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_text_document \
	  --vocab-file ${DATAPATH}/bpe/gpt2-vocab.json \
	  --merge-file ${DATAPATH}/bpe/gpt2-merges.txt \
	  --split 969,30,1 \
	  --eval-iters 100 \
	  --log-interval 5 \
	  --eval-interval 1000 \
	  --save-interval 10000 \
	  --clip-grad 1.0 \
	  --fp16 \
	  --DDP-impl local \
	  --loss-scale 8192 \
	  --checkpoint-activations "

run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_gpt.py ${options}"


srun -l \
     --container-image "nvcr.io#nvidia/pytorch:21.05-py3" \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"
