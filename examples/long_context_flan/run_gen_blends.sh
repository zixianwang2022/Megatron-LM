# model_name=flanv1_gpt3-43b-multi-1.1t-gtc-itp-16k-tian_43b_64_3e-7
# step=9000
# model_name=flanv1_gpt3-43b-multi-1.1t-gtc-base_43b_64_3e-7
# step=22500
model_name=qc_gpt3-43b-multi-1.1t-gtc-base_43b_128_5e-6
step=1000
# model_name=qc_gpt3-43b-multi-1.1t-gtc-itp-16k-tian_43b_128_5e-6
# step=1000
# model_name=qc_gpt3-43b-multi-1.1t-gtc-itp-32k-tian-tp16pp4_43b_128_5e-6
# step=500


# model_name=qa_blendv12_gpt_1e-8_conv_quiet_cockatoo_pp1_fixed_doc2dial_same_format_ctx1_43b_64_3e-7
# step=3000

num_ctxs=5
# echo `ls /lustre/fsw/adlr/adlr-nlp/pengx/inform-retriever/code/scroll_eval_data`
# for task in gov_report.dragon_retriever_chunkbysents300 narrative_qa.dragon_retriever_chunkbysents300 qasper.dragon_retriever_chunkbysents300 qmsum.dragon_retriever_chunkbysents300 quality.dragon_retriever_chunkbysents300 summ_screen_fd.dragon_retriever_chunkbysents300
# do
#     # echo $task
#     echo bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh $task 43b greedy test 0 200 \$step \$num_ctxs \$model_name true
#     echo bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh $task 43b greedy test 0 200 \$step \$num_ctxs \$model_name
# done


bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh gov_report.dragon_retriever_chunkbysents300 43b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.dragon_retriever_chunkbysents300 43b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.dragon_retriever_chunkbysents300 43b greedy test 0 2000 $step $num_ctxs $model_name
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh qasper.dragon_retriever_chunkbysents300 43b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh qasper.dragon_retriever_chunkbysents300 43b greedy test 0 2000 $step $num_ctxs $model_name
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh qmsum.dragon_retriever_chunkbysents300 43b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh qmsum.dragon_retriever_chunkbysents300 43b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh quality.dragon_retriever_chunkbysents300 43b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh quality.dragon_retriever_chunkbysents300 43b greedy test 0 2000 $step $num_ctxs $model_name
bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh summ_screen_fd.dragon_retriever_chunkbysents300 43b greedy test 0 200 $step $num_ctxs $model_name
