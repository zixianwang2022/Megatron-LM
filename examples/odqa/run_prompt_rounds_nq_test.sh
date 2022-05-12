#!/bin/bash

export DATANAME='nq-test'


# export EXPNAME='1.3b_gc_357m_ans'
# gpu='0 1 2 3 4 5 6 7'
# # rnds='1 2 3 4 5 6 7 8'
# rnds='9 10 11 12 13 14 15 16'


### this is for top1 + 357m ans model
export EXPNAME='top1_ctx_357m_ans_reversed'
gpu='2'
rnds='1'


array1=($gpu)
array2=($rnds)

for i in `seq 1 ${#array1[@]}`
    do
        nohup sh examples/odqa/prompt_answer_generation_nq_arg_1.3B_357m.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        echo "round${i} finished!"
    done