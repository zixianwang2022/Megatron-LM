#!/bin/bash

export DATANAME='nq'
export EXPNAME='k10_357m_gc_357m_multisetdpr_queryctx_p0.9'

export gpu_id='5'

for i in `seq 59 59` 
    do
        nohup sh examples/odqa/prompt_answer_generation_nq_arg.sh $RANDOM rnd${i} ${gpu_id}> logs/prompt_answer_generation_${DATANAME}_${EXPNAME}_rnd${i}.txt &
        wait
        echo "round${i} finished!"
    done