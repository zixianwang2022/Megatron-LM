model_name=qc_llama2_text_7b_cont_4k_7b_64_2.0e-5_7b_128_5e-6_step_1000_bad_instruction_following
model_name=qc_llama2_text_7b_cont_4k_7b_64_2.0e-5_7b_128_5e-6_step_4000_bad_instruction_following
model_name=qc_llama2_text_7b_cont_4k_7b_64_2.0e-5_7b_128_5e-6_step_6000_bad_instruction_following
model_name=qc_llama2_text_7b_itp-16k_7b_64_2.0e-5_7b_128_5e-6_step_1000_bad_instruction_following
model_name=qc_llama2_text_7b_itp-32k_7b_64_2.0e-5_7b_128_5e-6_step_1000
model_name=qc_llama2_text_7b_itp-32k_7b_64_2.0e-5_7b_128_5e-6_step_500
step=416

num_ctxs=5
retriever=dragon_retriever_chunkbysents300

bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.${retriever} 7b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.${retriever} 7b greedy test 0 500 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.${retriever} 7b greedy test 500 500 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.${retriever} 7b greedy test 1000 500 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh narrative_qa.${retriever} 7b greedy test 1500 500 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qasper.${retriever} 7b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qasper.${retriever} 7b greedy test 0 2000 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qmsum.${retriever} 7b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh qmsum.${retriever} 7b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quality.${retriever} 7b greedy test 0 2000 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh quality.${retriever} 7b greedy test 0 2000 $step $num_ctxs $model_name

bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh musique.${retriever} 7b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh musique.${retriever} 7b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh hotpotqa.${retriever} 7b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh hotpotqa.${retriever} 7b greedy test 0 200 $step $num_ctxs $model_name
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh multifieldqa_en.${retriever} 7b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft.sh multifieldqa_en.${retriever} 7b greedy test 0 200 $step $num_ctxs $model_name
