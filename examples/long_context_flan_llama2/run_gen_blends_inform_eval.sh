# model_name=qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6
# model_name=long_qc_v3_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_step_1000
# model_name=long_qc_v1_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_step_1000
# model_name=multiturn_qa_blend_commercial_v5_qc_llama2_text_70b_itp-32k_step_500
# model_name=multiturn_qa_blend_v2_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_128_5e-6_step_500
# model_name=multiturn_qa_blend_commercial_v5_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_128_5e-6_step_500
model_name=multiturn_qa_blend_v2_qc_llama2_text_70b_itp-32k_step_500
step=4000
# model_name=long_fqa_research_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_64_3e-7_step_500
# step=4500
model_name=primitive_stingray16k_fqa_qc_llama2_text_70b_itp-32k_70b_128_1.0e-5_70b_128_5e-6_70b_64_3e-7_step_500
step=3000
num_ctxs=5

bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh nq 70b greedy test  0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test  0 250 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test  0 250 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh nv_benefits_dragon_retriever300_retrieved_generic 70b greedy test 0 250 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test 0 250 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 $step $num_ctxs $model_name true

bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh hotpotqa 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh summ_screen_fd 70b greedy test 0 200 $step $num_ctxs $model_name true

bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh musique 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh multifieldqa_en 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh qasper 70b greedy test 0 200 $step $num_ctxs $model_name true
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh qmsum 70b greedy test 0 200 $step $num_ctxs $model_name true


## multi-turn qa
# doc2dial 3939 samples
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh doc2dial 70b greedy test 0 1000 $step $num_ctxs $model_name true
# quac 7354 samples
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh quac 70b greedy test 0 1000 $step $num_ctxs $model_name true
# qrecc 2805 samples
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh qrecc 70b greedy test 0 1000 $step $num_ctxs $model_name true
# sharc 10000 samples
bash examples/long_context_flan_llama2/generate_multijob_ckpt_step_cross_sft_inform.sh sharc 70b greedy test 0 1000 $step $num_ctxs $model_name true

