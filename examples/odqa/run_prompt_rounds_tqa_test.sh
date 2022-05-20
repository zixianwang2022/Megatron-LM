#!/bin/bash

# export DATANAME='tqa-test'
# export EXPNAME='k10_1.3b_gc_357m_ans'
# export EXPNAME='357m_gc_357m_ans'
# gpu='0 1 2 3 4 5 6 7'
# rnds='1 2 3 4 5 6 7 8'
# rnds='9 10 11 12 13 14 15 16'



### this is for the top1 + 1.3B ans model
# export EXPNAME='top1_ctx_1.3B_ans_reversed'
# gpu='1'
# rnds='1'

### this is for the golden + 1.3B ans model
# export EXPNAME='golden_ctx_1.3B_ans'
# gpu='1'
# rnds='1'

### this is for the top1 + 1.3B ans model
# export EXPNAME='top1_ctx_1.3b_ans_reversed'
# gpu='0'
# rnds='1'

### this is for the topk + 1.3B ans model
# forgot the dataname
export EXPNAME='topk_ctx_1.3b_ans'
gpu='1 2 3 4 5 6 7'
topk='2 3 4 5 6 7 8'


## the retrieval topk + 357m model
# export EXPNAME='topk_ctx_357m_ans'
# gpu='1 2 3 4 5 6 7'
# topk='2 3 4 5 6 7 8'

# export DATANAME='tqa-test'
# export EXPNAME='357m_gc_357m_ans_beam'
# gpu='0 1 2 3 4 5 6 7'
# rnds='1 2 3 4 5 6 7 8'
# rnds='9 10 11 12 13 14 15 16'



array1=($gpu)
# array2=($rnds)
array2=($topk)

for i in `seq 1 ${#array1[@]}`
    do
        # nohup sh examples/odqa/prompt_answer_generation_tqa_arg_1.3B_357m.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_tqa_arg.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_tqa_arg_1.3B.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_tqa_arg.sh $RANDOM ${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_top${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_tqa_arg.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        nohup sh examples/odqa/prompt_answer_generation_tqa_arg_1.3B.sh $RANDOM ${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_top${array2[$i-1]}.txt &


        echo "round${i} finished!"
    done