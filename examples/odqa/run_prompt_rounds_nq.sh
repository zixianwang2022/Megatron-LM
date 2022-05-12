#!/bin/bash

export DATANAME='nq_test'
# export EXPNAME='api_530_gc_357m_ans'
# export EXPNAME='api_530b_gc_1.3b_ans'

# gpu='0 1 2 3 4 5 6 7'
# rnds='1 2 3 4 5 6 7 8'
# rnds='9 10 11 12 13 14 15 16'

## this is just for golden + 1.3 B ans
# export EXPNAME='golden_ctx_1.3b_ans'

## 
# export EXPNAME='top1_ctx_1.3b_ans_reversed'
# gpu='0'
# rnds='1'

## the new 357m + 357m
# export EXPNAME='new_357m_gc_357m_ans'
# gpu='0 1 2 3 4 5 6 7'
# rnds='1 2 3 4 5 6 7 8'


## the retrieval topk + 357m model
export EXPNAME='topk_ctx_357m_ans'
# gpu='2 3 4 5 6 7'
# topk='2 3 4 5 6 7'
gpu='0'
topk='8'


array1=($gpu)
# array2=($rnds)
array2=($topk)

for i in `seq 1 ${#array1[@]}`
    do
        nohup sh examples/odqa/prompt_answer_generation_nq_arg.sh $RANDOM ${array2[$i-1]} ${array1[$i-1]}> logs/api_prompt_answer_generation_${DATANAME}_${EXPNAME}_top${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_nq_arg_1.3B.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]}> logs/api_prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        echo "round${i} finished!"
    done