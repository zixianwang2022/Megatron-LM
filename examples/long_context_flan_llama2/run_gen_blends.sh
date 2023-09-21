# model_name=qc_llama2_text_70b_base_70b_128_5e-6
# model_name=qc_llama2_text_70b_itp-16k_70b_128_1.0e-5_70b_128_5e-6
# model_name=qc_llama2_text_70b_base_70b_128_5e-6
model_name=qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6
step=1000
# model_name=qc_llama2_text_70b_base_70b_64_2e-5
# step=4000

num_ctxs=10
# echo `ls /lustre/fsw/adlr/adlr-nlp/pengx/inform-retriever/code/scroll_eval_data`
# for task in gov_report.dragon_retriever_chunkbysents300 narrative_qa.dragon_retriever_chunkbysents300 qasper.dragon_retriever_chunkbysents300 qmsum.dragon_retriever_chunkbysents300 quality.dragon_retriever_chunkbysents300 summ_screen_fd.dragon_retriever_chunkbysents300
# do
#     # echo $task
#     echo bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh $task 70b greedy test 0 200 $step $num_ctxs $model_name true
#     echo bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh $task 70b greedy test 0 200 $step $num_ctxs $model_name
# done

bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh gov_report.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh gov_report.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.dragon_retriever_chunkbysents300 70b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.dragon_retriever_chunkbysents300 70b greedy test 0 1000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.dragon_retriever_chunkbysents300 70b greedy test 1000 1000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qasper.dragon_retriever_chunkbysents300 70b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qasper.dragon_retriever_chunkbysents300 70b greedy test 0 2000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qmsum.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qmsum.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quality.dragon_retriever_chunkbysents300 70b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quality.dragon_retriever_chunkbysents300 70b greedy test 0 2000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh summ_screen_fd.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh summ_screen_fd.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name


bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh musique.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh musique.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh hotpotqa.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh hotpotqa.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh multifieldqa_en.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh multifieldqa_en.dragon_retriever_chunkbysents300 70b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh doc2dial_full_dialogue.dragon_retriever_chunkbysents300 70b greedy test 0 1000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh doc2dial_full_dialogue.dragon_retriever_chunkbysents300 70b greedy test 0 500 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh doc2dial_full_dialogue.dragon_retriever_chunkbysents300 70b greedy test 500 500 $step $num_ctxs $model_name
