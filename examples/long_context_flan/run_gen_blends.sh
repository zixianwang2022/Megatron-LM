# model_name=flanv1_gpt3-43b-multi-1.1t-gtc-itp-16k-tian_43b_64_3e-7
# step=9000
# model_name=flanv1_gpt3-43b-multi-1.1t-gtc-base_43b_64_3e-7
# step=22500
# model_name=qc_gpt3-43b-multi-1.1t-gtc-base_43b_128_5e-6
# step=1000
# model_name=qc_gpt3-43b-multi-1.1t-gtc-itp-16k-tian_43b_128_5e-6
# step=1000
model_name=qc_gpt3-43b-multi-1.1t-gtc-itp-32k-tian-tp16pp4_43b_128_5e-6
step=500
num_ctxs=5

for task in `ls /lustre/fsw/adlr/adlr-nlp/pengx/inform-retriever/code/scroll_eval_data`
do
    # echo $task
    echo bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh $task 43b greedy test 0 200 $step $num_ctxs $model_name true
    echo bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh $task 43b greedy test 0 200 $step $num_ctxs $model_name
done
