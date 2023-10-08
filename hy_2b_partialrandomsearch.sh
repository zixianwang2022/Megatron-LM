#!/bin/bash

DIR=`pwd`

# lr_array=( 0 1 2 3 4 5 6 7 8 9 10 )
# lr_array=( 0 1 2 3 4 5 )
lr_array=( 0 )
# init_scale=2.5
# emb_multiplier=10.0
init_scale=1.0
emb_multiplier=1.0
# base_shape_hz=256
base_shape_hz=2048

for i in ${lr_array[@]};
do
    lr=$(bc -l <<< "0.0002*2^$i")
    min_lr=$(bc -l <<< "0.00002*2^$i")
    echo $lr
    sed "s/BASE_SHAPE_HZ/${base_shape_hz}/g;s/LEARNING_RATE/${lr}/g;s/MIN_LR/${min_lr}/g;s/INIT_SCALE/${init_scale}/g;s/EMB_MULTIPLIER/${emb_multiplier}/g;s/POS_MULTIPLIER/${pos_multiplier}/g" gpt3_2b_gtc_llr-eng-cc202240_lr_init_emb_template_0.sh > $DIR/gpt3_2b_gtc_llr-eng-cc202240_template_lr_init_emb_pos_run_base${base_shape_hz}.sh
    sbatch gpt3_2b_gtc_llr-eng-cc202240_template_lr_init_emb_pos_run_base${base_shape_hz}.sh
    sleep 5
done

