# model_name=qc_llama2_text_70b_base_70b_128_5e-6
# model_name=qc_llama2_text_70b_itp-16k_70b_128_1.0e-5_70b_128_5e-6
# model_name=qc_llama2_text_70b_base_70b_128_5e-6
model_name=qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6
# model_name=qc_llama2_text_70b_cont_4k_70b_128_1.0e-5_70b_128_5e-6_step_1000
# model_name=qc_llama2_text_70b_cont_4k_70b_128_1.0e-5_70b_128_5e-6_step_4000
# model_name=qc_llama2_text_70b_cont_4k_70b_128_1.0e-5_70b_128_1.0e-5_70b_128_5e-6_step_2000
# model_name=qc_llama2_text_70b_cont_4k_70b_128_1.0e-5_70b_128_1.0e-5_70b_128_5e-6_step_4000
# model_name=qc_llama2_text_70b_cont_4k_70b_128_1.0e-5_70b_128_5e-6_step_2500
step=500
# model_name=qc_llama2_text_70b_base_70b_64_2e-5
# step=4000

# model_name=multiturn_qa_blend_v2_qc_llama2_text_70b_itp-32k_step_500
# step=4000
# model_name=long_fqa_research_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_64_3e-7_step_500
# step=4500

num_ctxs=5

# for i in {1..8}
# do
#     num_gen=500
#     start_idx=$((i*num_gen-num_gen))
#     bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh doc2dial_long 70b greedy test ${start_idx} ${num_gen} $step $num_ctxs $model_name true
# done
# for i in {1..10}
# do
#     num_gen=750
#     start_idx=$((i*num_gen-num_gen))
#     bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quac_long 70b greedy test ${start_idx} ${num_gen} $step $num_ctxs $model_name true
# done
# for i in {1..4}
# do
#     num_gen=750
#     start_idx=$((i*num_gen-num_gen))
#     bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qrecc_long 70b greedy test ${start_idx} ${num_gen} $step $num_ctxs $model_name true
# done

# retriever=dragon_retriever_chunkbysents300
retriever=e5_mistral_retriever_chunkbysents300
# echo `ls /lustre/fsw/adlr/adlr-nlp/pengx/inform-retriever/code/scroll_eval_data`
# for task in gov_report.${retriever} narrative_qa.${retriever} qasper.${retriever} qmsum.${retriever} quality.${retriever} summ_screen_fd.${retriever}
# do
#     # echo $task
#     echo bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh $task 70b greedy test 0 200 $step $num_ctxs $model_name true
#     echo bash examples/long_context_flan/generate_multijob_ckpt_step_cross_sft.sh $task 70b greedy test 0 200 $step $num_ctxs $model_name
# done

# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh gov_report.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name true
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh gov_report.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.${retriever} 70b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.${retriever} 70b greedy test 0 1000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.${retriever} 70b greedy test 1000 1000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qasper.${retriever} 70b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qasper.${retriever} 70b greedy test 0 2000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qmsum.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qmsum.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quality.${retriever} 70b greedy test 0 1000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quality.${retriever} 70b greedy test 1000 1000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quality.${retriever} 70b greedy test 0 1000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quality.${retriever} 70b greedy test 1000 1000 $step $num_ctxs $model_name
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh summ_screen_fd.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name true
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh summ_screen_fd.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name


# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh musique.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name true
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh musique.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh hotpotqa.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name true
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh hotpotqa.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh multifieldqa_en.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name true
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh multifieldqa_en.${retriever} 70b greedy test 0 200 $step $num_ctxs $model_name
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh doc2dial_full_dialogue.${retriever} 70b greedy test 0 1000 $step $num_ctxs $model_name true
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh doc2dial_full_dialogue.${retriever} 70b greedy test 0 500 $step $num_ctxs $model_name
# bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh doc2dial_full_dialogue.${retriever} 70b greedy test 500 500 $step $num_ctxs $model_name
