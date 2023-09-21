
# model_name=multiturn_qa_blend_commercial_v5_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7
# model_name=multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_5e-6
model_name=multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_1e-6
num_ctxs=5

## single-turn qa (batch-1)
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nq 70b greedy test  0 200 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test  0 250 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test  0 250 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh nv_benefits_dragon_retriever300_retrieved_generic 70b greedy test 0 250 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test 0 250 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 3000 $num_ctxs $model_name true

## single-turn-qa (batch-2)
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh BioASQ 70b greedy test 0 1000 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh DuoRC_ParaphraseRC 70b greedy test 0 1000 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh boolq 70b greedy test 0 1000 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh msmarco 70b greedy test 0 1000 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh multirc 70b greedy test 0 1000 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh race 70b greedy test 0 1000 3000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh TextbookQA 70b greedy test 0 1000 3000 $num_ctxs $model_name true

## multi-turn qa
# doc2dial 3939 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh doc2dial 70b greedy test 0 1000 3000 $num_ctxs $model_name true
# quac 7354 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh quac 70b greedy test 0 1000 3000 $num_ctxs $model_name true
# qrecc 2805 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh qrecc 70b greedy test 0 1000 3000 $num_ctxs $model_name true
# sharc 10000 samples
bash examples/fqa_llama2/generate_llama2_fqa_zeroshot.sh sharc 70b greedy test 0 1000 3000 $num_ctxs $model_name true
