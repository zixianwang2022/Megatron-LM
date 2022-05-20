#!/bin/bash

# export DATANAME='wq-test-new'
# export EXPNAME='1.3b_gc_357m_ans'
# gpu='0 1 2 3 4 5 6 7'
# rnds='1 2 3 4 5 6 7 8'
# rnds='9 10 11 12 13 14 15 16'
# rnds='18 19 20 21 22 23 24 25'
# rnds='26 27 28 29 30 31 32'

# export DATANAME='wq-test-new'
# export EXPNAME='530b_gc_1.3b_ans'
# gpu='0 1 2 3 4 5 6'
# rnds='2 3 4 5 6 7 8'
# gpu='2'
# rnds='1'


export DATANAME='wq-test-new'
export EXPNAME='530b_gc_357m_ans'
gpu='0 1 2 3 4 5 6'
rnds='2 3 4 5 6 7 8'

# export DATANAME='wq-test-new'
# export EXPNAME='golden_ctx_1.3b_ans'
# gpu='3'
# rnds='1'

# export EXPNAME='top1_ctx_1.3b_ans_reversed'
# gpu='4'
# rnds='1'


# the retrieval topk + 357m model
# export EXPNAME='topk_ctx_357m_ans'
# gpu='1 2 3 4 5 6 7'
# topk='2 3 4 5 6 7 8'

## the topk + 1.3B model
# export EXPNAME='topk_ctx_1.3b_ans'
# gpu='1 2 3 4 5 6 7'
# topk='2 3 4 5 6 7 8'



array1=($gpu)
array2=($rnds)
# array2=($topk)

for i in `seq 1 ${#array1[@]}`
    do
        # nohup sh examples/odqa/prompt_answer_generation_wq_arg_1.3B_357m.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]} > logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_wq_arg_1.3B.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]} > logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_wq.sh $RANDOM ${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_wq_orig.sh $RANDOM ${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_top${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_wq_arg_1.3B.sh $RANDOM ${array2[$i-1]} ${array1[$i-1]} > logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_top${array2[$i-1]}.txt &
        nohup sh examples/odqa/prompt_answer_generation_wq_orig.sh $RANDOM rnd${array2[$i-1]} ${array1[$i-1]}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${array2[$i-1]}.txt &

        echo "round${i} finished!"
    done