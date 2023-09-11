
model_name=llama2_chat_70b
# model_name=llama2_text_70b_with_qc
num_ctxs=5

## single-turn qa (batch-1)
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh nq 70b greedy test  0 200 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test  0 250 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test  0 250 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh nv_benefits_dragon_retriever300_retrieved_generic 70b greedy test 0 250 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test 0 250 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 $num_ctxs $model_name true

# ## single-turn qa (batch-2)
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh BioASQ 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh DuoRC_ParaphraseRC 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh boolq 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh msmarco 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh multirc 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh race 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh TextbookQA 70b greedy test 0 1000 $num_ctxs $model_name true

## multi-turn qa
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 70b greedy test 0 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 70b greedy test 0 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 70b greedy test 0 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sharc 70b greedy test 0 1000 $num_ctxs $model_name true
