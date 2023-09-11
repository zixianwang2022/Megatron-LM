
# model_name=malachite_sawfly
# model_name=neat_spoonbill
# model_name=vehement_coyote
model_name=wine_jackdaw
num_ctxs=5

## single-turn qa (batch-1)
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh nq 43b greedy test  0 200 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 $num_ctxs $model_name true

## single-turn qa (batch-2)
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh BioASQ 43b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh DuoRC_ParaphraseRC 43b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh boolq 43b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh msmarco 43b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh multirc 43b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh race 43b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh TextbookQA 43b greedy test 0 1000 $num_ctxs $model_name true

## multi-turn qa
# doc2dial 3939 samples
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh doc2dial 43b greedy test 0 1000 $num_ctxs $model_name true
# quac 7354 samples
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh quac 43b greedy test 0 1000 $num_ctxs $model_name true
# qrecc 2805 samples
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh qrecc 43b greedy test 0 1000 $num_ctxs $model_name true
# sharc 10000 samples
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_others_conv.sh sharc 43b greedy test 0 1000 $num_ctxs $model_name true
