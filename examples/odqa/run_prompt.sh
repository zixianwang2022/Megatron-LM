#!/bin/bash

export DATANAME='tqa'
export EXPNAME='api_357m'

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
        nohup sh examples/odqa/prompt_answer_generation_arg.sh ${array1[$i-1]} ${array2[$i-1]} $RANDOM > logs/api_answer_generation_${DATANAME}_${EXPNAME}_${array2[$i-1]}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_arg.sh ${array1[$i-1]} ${array2[$i-1]} $RANDOM $1 > logs/answer_generation_${DATANAME}_${EXPNAME}_${array2[$i-1]}_$1_withprob.txt &
    done


wait
echo '============='  
# wc /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_$1_$2.top2.txt 
mv /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/api/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_${list}_withprob.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/api/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_${list}_withprob.txt
# cat /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_2000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_4000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_6000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_8000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_10000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_11313_$2.txt > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_$2.txt   
# cat /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_2000.top2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_4000.top2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_6000.top2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_8000.top2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_10000.top2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_11313.top2.txt > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all.top2.txt   
echo 'concatenation done!'
