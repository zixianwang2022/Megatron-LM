#!/bin/bash

#SBATCH -p batch_block1,batch_block2
#SBATCH --nodes=1
#SBATCH -A adlr
#SBATCH -t 0:30:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:flash-off
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton

# standard_tasks="abstract_narrative_understanding general_knowledge human_organs_senses intent_recognition riddle_sense similarities_abstraction simple_arithmetic_json simple_arithmetic_json_multiple_choice undo_permutation unit_conversion qa_wikidata linguistic_mappings date_understanding conlang_translation"
# tydiqa_tasks="tydiqa_goldp.ar tydiqa_goldp.bn tydiqa_goldp.en tydiqa_goldp.fi tydiqa_goldp.id tydiqa_goldp.ko tydiqa_goldp.ru tydiqa_goldp.sw tydiqa_goldp.te"
standard_tasks="general_knowledge"
tydiqa_tasks="tydiqa_goldp.ar"

task_lists="standard_tasks tydiqa_tasks"
declare -A my_dict
my_dict["standard_tasks"]=" --max_length=64 --json_shots='0,1,2' "
my_dict["tydiqa_tasks"]=" --max_length=16 --json_shots='1,4' "

model_name_list="gpt3-43b-multi-1.1t-gtc" 
for model_name in $model_name_list; do
  for tasks in $task_lists; do
    for task in ${!tasks}; do
        sbatch --export=TASK_ARGS=" --task=${task} ${my_dict[$tasks]}",MODEL_NAME="${model_name}" \
        --output={LOG_DIRECTORY}/log-megatron-${model_name}-bigbench-${task}.out \
        --job-name=swdl-big_nlp:megatron-${model_name}-bigbench-${task} \
        {JOB_SPECIFICATION_SCRIPT}.sh
    done
  done
done

# eof
