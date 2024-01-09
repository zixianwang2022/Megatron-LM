
# model_name=multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blendv2_llama2_chat_70b_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blendv2_llama2_chat_70b_multiturn_same_format_ctx1_70b_64_6e-7
# model_name=multiturn_qa_blend_commercial_v9_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_finance_v1_llama2_chat_70b_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v10_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v12_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_finance_v2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7

# model_name=multiturn_qa_blendv5_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blendv6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v15_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7

# model_name=sft_blend_llama2_text_70b_same_format_ctx1_70b_128_5e-6

# model_name=multiturn_qa_blendv7_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7

## financial
# model_name=multiturn_qa_blend_finance_v1_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v3_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v4_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7

## latest checkpoints
# model_name=multiturn_qa_blend_commercial_v19_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
model_name=multiturn_qa_blend_commercial_v19_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7


num_ctxs=5

# ## nvolve retrieval results
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh att_nvolve 70b greedy test  0 200 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh iternal_nvolve 70b greedy test  0 250 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nvit_nvolve 70b greedy test  0 250 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sandia_nvolve 70b greedy test  0 250 3442 $num_ctxs $model_name true

# ## table data
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh fetaqa 70b greedy test 0 1001 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh WikiTableQuestions 70b greedy test 0 2200 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh WikiTableQuestions 70b greedy test 2200 2200 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh convfinqav3 70b greedy test 0 1500 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh finqav2 70b greedy test 0 1500 3442 $num_ctxs $model_name true

# ## financial
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh tatqav2 70b greedy test 0 1000 3442 $num_ctxs $model_name true
### bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh finqav2 70b greedy test 0 1000 3442 $num_ctxs $model_name true
### bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh convfinqav2 70b greedy test 0 1000 3442 $num_ctxs $model_name true

# # ## single-turn qa (batch-1)
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nq 70b greedy test  0 200 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test  0 250 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test  0 250 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nv_benefits_dragon_retriever300_retrieved_generic 70b greedy test 0 250 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test 0 250 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 3435 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sandia 70b greedy test 0 250 3435 $num_ctxs $model_name true


# ## single-turn-qa (batch-2)
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh BioASQ 70b greedy test 0 1000 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh DuoRC_ParaphraseRC 70b greedy test 0 1000 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh boolq 70b greedy test 0 1000 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh msmarco 70b greedy test 0 1000 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh multirc 70b greedy test 0 1000 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh race 70b greedy test 0 1000 3442 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh TextbookQA 70b greedy test 0 1000 3442 $num_ctxs $model_name true

# ## multi-turn qa
# # doc2dial 3939 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 70b greedy test 0 1000 3442 $num_ctxs $model_name true
# # quac 7354 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 70b greedy test 0 1000 3442 $num_ctxs $model_name true
# # qrecc 2805 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 70b greedy test 0 1000 3442 $num_ctxs $model_name true

# # sharc 34310 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sharc 70b greedy test 0 1000 3442 $num_ctxs $model_name true




## single-turn qa (batch-1) 13B
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nq 13b greedy test  0 200 3600 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 13b greedy test  0 250 3600 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 13b greedy test  0 250 3600 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 13b greedy test 0 250 3600 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nv_benefits_dragon_retriever300_retrieved_generic 13b greedy test 0 250 3600 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 13b greedy test 0 250 3600 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 13b greedy test 0 250 3600 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sandia 13b greedy test 0 250 3600 $num_ctxs $model_name true

