
model_name=multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
num_ctxs=5

## single-turn qa (batch-1)
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true

# ## single-turn-qa (batch-2)
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh BioASQ 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh DuoRC_ParaphraseRC 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh boolq 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh msmarco 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh multirc 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh race 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh TextbookQA 43b greedy test 0 1000 3000 $num_ctxs $model_name true

# ## multi-turn qa
# # doc2dial 3939 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# # quac 7354 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# # qrecc 2805 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 43b greedy test 0 1000 3000 $num_ctxs $model_name true
# # sharc 10000 samples
# bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sharc 43b greedy test 0 1000 3000 $num_ctxs $model_name true
