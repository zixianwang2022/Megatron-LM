
model_name=multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_fewshot_ctx1_70b_64_3e-7
num_ctxs=5

## single-turn qa (batch-1)
bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh nq 70b greedy test  0 200 3600 $num_ctxs $model_name true 3
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test  0 250 3600 $num_ctxs $model_name true 3
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test  0 250 3600 $num_ctxs $model_name true 3
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 3600 $num_ctxs $model_name true 3
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh nv_benefits_dragon_retriever300_retrieved_generic 70b greedy test 0 250 3600 $num_ctxs $model_name true 3
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test 0 250 3600 $num_ctxs $model_name true 3
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 3600 $num_ctxs $model_name true 3

## single-turn qa (batch-2)
bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh BioASQ 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh DuoRC_ParaphraseRC 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh boolq 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh msmarco 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh multirc 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh race 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh TextbookQA 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3

# ## multi-turn qa
# # doc2dial 3939 samples
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh doc2dial 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
# # quac 7354 samples
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh quac 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
# # qrecc 2805 samples
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh qrecc 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
# # sharc 10000 samples
# bash examples/fqa_llama2/generate_llama2_fqa_fewshot.sh sharc 70b greedy test 0 1000 3600 $num_ctxs $model_name true 3
