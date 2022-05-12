#!/bin/bash

export DATANAME='train_tqa'
export EXPNAME='k10_357m_gc_357m_multisetdpr_queryctx_p0.9'

list='1 2 3 4 5 6 7 8'
gpu='0 1 2 3 4 5 6 7'

# list='9 10'
# gpu='0 1'
# list='9'
# gpu='7'



array1=($gpu)
array2=($list)

echo "====================== new round $1 start!"
echo $1

for i in `seq 1 ${#array1[@]}`
    do
        nohup bash examples/odqa/prompt_answer_generation_arg_train.sh ${array1[$i-1]} ${array2[$i-1]} $RANDOM $1 > logs/answer_generation_${DATANAME}_${EXPNAME}_${array2[$i-1]}_$1_withprob.txt &
    done

wait
echo '=============' 

# CAT_LIST=""
# for i in {1..10..1}
#     do
#     CAT_LIST="${CAT_LIST} /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_${i}_$1_withprob.txt"
#     done

# CMD="cat${CAT_LIST} > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_$1_withprob.txt"
# eval $CMD

# lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_$1_withprob.txt | awk {'print $1'})

# if [ "$lines" -eq "78785" ]; then
#             echo " concatenation in $1 successful!"
#         else
#             echo "concatenation in $1 failed!"
#         fi
