
# model_name=multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blendv1_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blendv1_1_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blendv2_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7
# model_name=multiturn_qa_blend_finance_v6_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7
# model_name=multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v19_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v19_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v19_1_llama2_chat_13b_multiturn_same_format_ctx1_13b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v23_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7

# model_name=multiturn_qa_blend_finance_v6_1_llama2_text_7b_with_qc_multiturn_same_format_ctx1_7b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v23_1_llama2_text_7b_with_qc_multiturn_same_format_ctx1_7b_64_3e-7

# model_name=multiturn_qa_blendv2_llama2_chat_70b_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v3_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v4_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v4_1_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v5_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v6_nosingle_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v6_llama2_chat_70b_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v6_llama2_text_70b_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v7_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7_v1
# model_name=multiturn_qa_blend_finance_v7_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v7_llama2_text_70b_with_qc_multiturn_promptv2_same_format_ctx1_70b_64_3e-7

# model_name=multiturn_qa_blend_commercial_v19_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v19_llama2_chat_70b_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v20_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v21_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v22_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v23_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
model_name=multiturn_qa_blend_commercial_v24_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7

# model_name=sft_blend_llama2_text_13b_same_format_ctx1_13b_128_5e-6

num_ctxs=5
# num_ctxs=10
# num_ctxs=3

## convfinqav3 1490 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh convfinqav3 70b greedy test 0 1500 3445 $num_ctxs $model_name true

## sqa 3100 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sqa 70b greedy test 0 1500 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sqa 70b greedy test 1500 1600 3445 $num_ctxs $model_name true

## doc2dial 3939 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 70b greedy test 0 2000 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 70b greedy test 2000 2000 3445 $num_ctxs $model_name true

## quac 7354 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 70b greedy test 0 2000 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 70b greedy test 2000 2000 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 70b greedy test 4000 2000 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 70b greedy test 6000 2000 3445 $num_ctxs $model_name true

## qrecc 2805 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 70b greedy test 0 2000 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 70b greedy test 2000 2000 3445 $num_ctxs $model_name true

## coqa 7983 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 70b greedy test 0 2000 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 70b greedy test 2000 2000 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 70b greedy test 4000 2000 3445 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 70b greedy test 6000 2000 3445 $num_ctxs $model_name true

## doqa_cooking 1797 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_cooking 70b greedy test 0 2000 3445 $num_ctxs $model_name true

## doqa_movies 1884 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_movies 70b greedy test 0 2000 3445 $num_ctxs $model_name true

## doqa_travel 1713 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_travel 70b greedy test 0 2000 3445 $num_ctxs $model_name true

## topiocqa 2514 samples
# num_ctxs = 20
# num_ctxs = 30
# num_ctxs = 12
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa 70b greedy test 0 1300 3445 20 $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa 70b greedy test 1300 1300 3445 20 $model_name true

## hybriddial 1111 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh hybriddial 70b greedy test 0 1200 3445 $num_ctxs $model_name true

## inscit 502 samples
## num_ctxs=20
## num_ctxs=30
## num_ctxs=12
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh inscit 70b greedy test 0 550 3445 20 $model_name true


# ## llmware
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh llmware 70b greedy test 0 500 3442 $num_ctxs $model_name true
# # llmware_unanswerable
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh llmware_unanswerable 70b greedy test 0 500 3442 $num_ctxs $model_name true


# # HybridQA 3466 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh HybridQA 70b greedy test 0 1500 3435 $num_ctxs $model_name true

# ## finqav2 1147 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh finqav2 70b greedy test 0 1500 3435 $num_ctxs $model_name true


# ### compared to original dragon retrieval
# ## doc2dial_dragon 3939 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial_dragon 70b greedy test 0 2000 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial_dragon 70b greedy test 2000 2000 3435 $num_ctxs $model_name true

# ## quac_dragon 7354 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac_dragon 70b greedy test 0 2000 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac_dragon 70b greedy test 2000 2000 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac_dragon 70b greedy test 4000 2000 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac_dragon 70b greedy test 6000 2000 3435 $num_ctxs $model_name true

# # qrecc_dragon 2805 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc_dragon 70b greedy test 0 2000 3435 10 $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc_dragon 70b greedy test 2000 2000 3435 10 $model_name true

# # topiocqa_dragon 2514 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa_dragon 70b greedy test 0 1300 3435 30 $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa_dragon 70b greedy test 1300 1300 3435 30 $model_name true

# # inscit_dragon 502 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh inscit_dragon 70b greedy test 0 550 3435 20 $model_name true



#### running for llama2 13b/7b ####

# ## doc2dial 3939 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 7b greedy test 0 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 7b greedy test 500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 7b greedy test 1000 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 7b greedy test 1500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 7b greedy test 2000 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 7b greedy test 2500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 7b greedy test 3000 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 7b greedy test 3500 500 3333 $num_ctxs $model_name true


# ## quac 7354 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 0 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 300 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 600 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 900 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 1200 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 1500 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 1800 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 2100 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 2400 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 2700 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 3000 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 3300 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 3600 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 3900 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 4200 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 4500 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 4800 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 5100 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 5400 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 5700 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 6000 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 6300 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 6600 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 6900 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 7b greedy test 7200 300 3333 $num_ctxs $model_name true


# ## qrecc 2805 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 0 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 300 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 600 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 900 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 1200 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 1500 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 1800 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 2100 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 2400 300 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 7b greedy test 2700 300 3333 $num_ctxs $model_name true


# ## doqa_cooking 1797 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_cooking 7b greedy test 0 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_cooking 7b greedy test 500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_cooking 7b greedy test 1000 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_cooking 7b greedy test 1500 500 3333 $num_ctxs $model_name true

# ## doqa_movies 1884 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_movies 7b greedy test 0 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_movies 7b greedy test 500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_movies 7b greedy test 1000 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_movies 7b greedy test 1500 500 3333 $num_ctxs $model_name true

# ## doqa_travel 1713 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_travel 7b greedy test 0 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_travel 7b greedy test 500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_travel 7b greedy test 1000 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doqa_travel 7b greedy test 1500 500 3333 $num_ctxs $model_name true

# ## topiocqa 2514 samples
# ## num_ctxs = 20
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa 7b greedy test 0 500 3333 20 $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa 7b greedy test 500 500 3333 20 $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa 7b greedy test 1000 500 3333 20 $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa 7b greedy test 1500 500 3333 20 $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh topiocqa 7b greedy test 2000 600 3333 20 $model_name true

# ## coqa 7983 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 7b greedy test 0 1000 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 7b greedy test 1000 1000 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 7b greedy test 2000 1000 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 7b greedy test 3000 1000 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 7b greedy test 4000 1000 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 7b greedy test 5000 1000 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 7b greedy test 6000 1000 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh coqa 7b greedy test 7000 1000 3333 $num_ctxs $model_name true

# ## convfinqav3 1490 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh convfinqav3 7b greedy test 0 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh convfinqav3 7b greedy test 500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh convfinqav3 7b greedy test 1000 500 3333 $num_ctxs $model_name true

# ## sqa 3100 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sqa 7b greedy test 0 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sqa 7b greedy test 500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sqa 7b greedy test 1000 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sqa 7b greedy test 1500 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sqa 7b greedy test 2000 500 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sqa 7b greedy test 2500 600 3333 $num_ctxs $model_name true

# ## hybriddial 1111 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh hybriddial 7b greedy test 0 600 3333 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh hybriddial 7b greedy test 600 600 3333 $num_ctxs $model_name true

# ## inscit 502 samples
# ## num_ctxs = 20
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh inscit 7b greedy test 0 550 3600 20 $model_name true





# # ## llmware 200 samples
# # bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh llmware 13b greedy test 0 500 3600 $num_ctxs $model_name true

