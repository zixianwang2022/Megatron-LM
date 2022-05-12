#!/bin/bash

export DATANAME='tqa_test'
export EXPNAME='530gc_1.3b_ans_splited'

list='2000 4000 6000 8000 10000 11313'
gpu='1 3 4 5 6 7'
# list='10000'
# gpu=$2

# list='11313'
# gpu='7'


array1=($gpu)
array2=($list)

# echo "====================== new round $2 start!"
# echo $1
# echo $2
# echo $2

# for i in `seq 1 ${#array1[@]}`
#     do
#         nohup sh examples/odqa/prompt_answer_generation_arg.sh ${array1[$i-1]} ${array2[$i-1]} $RANDOM > logs/qg_prompt_answer_generation_${DATANAME}_${EXPNAME}_${array2[$i-1]}.top2.txt &
#     done

for i in `seq 1 ${#array1[@]}`
    do
        nohup sh examples/odqa/prompt_answer_generation_tqa_splited_arg.sh ${array1[$i-1]} ${array2[$i-1]} $RANDOM > logs/api_answer_generation_${DATANAME}_${EXPNAME}_${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_arg.sh ${array1[$i-1]} ${array2[$i-1]} $RANDOM $1 > logs/answer_generation_${DATANAME}_${EXPNAME}_${array2[$i-1]}_$1_withprob.txt &
    done
