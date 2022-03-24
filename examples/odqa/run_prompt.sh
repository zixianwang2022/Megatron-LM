#!/bin/bash

export DATANAME='tqa'
export EXPNAME='k10_357m_gc_357m_multisetdpr_queryctx_p0.9'

list='2000 4000 6000 8000 10000 11313'
gpu='1 2 3 5 6 7'

# list='2000'
# gpu='1'

array1=($gpu)
array2=($list)

echo "====================== new round $2 start!"
echo $1
echo $2

for i in `seq 1 ${#array1[@]}`
    do
    # nohup echo ${array1[$i-1]} ${array2[$i-1]} $1 &
        nohup sh examples/odqa/prompt_answer_generation_arg.sh ${array1[$i-1]} ${array2[$i-1]} $1 $2 > logs/qg_prompt_answer_generation_${DATANAME}_${EXPNAME}_${array2[$i-1]}_$2.txt &
    done

wait 
echo '============='      
cat /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_2000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_4000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_6000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_8000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_10000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_11313_$2.txt > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_$2.txt   
echo 'concatenation done!'

# mv /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_2000_$2.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_2000_$2_test.txt