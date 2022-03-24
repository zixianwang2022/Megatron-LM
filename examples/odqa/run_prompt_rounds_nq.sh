#!/bin/bash

# random_seed='2345 3456 4567 5678 6789 7890'
export DATANAME='nq'
export EXPNAME='k10_357m_gc_357m_multisetdpr_queryctx_p0.9'

export gpu_id='0'

for i in `seq 61 63` 
    do
        # nohup sh examples/odqa/prompt_answer_generation.sh $RANDOM rnd${i} > logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${i}.txt &
        # nohup sh examples/odqa/prompt_answer_generation_2.sh $RANDOM rnd${i} > logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${i}.txt &
        nohup sh examples/odqa/prompt_answer_generation_3.sh $RANDOM rnd${i} ${gpu_id}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${i}.txt &
        wait
        echo "round${i} finished!"
    done