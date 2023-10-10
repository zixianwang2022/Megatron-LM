
model_name=quiet_cockatoo_8b_3.5t_5e_6_pp1
# model_name=quiet_cockatoo_8b_3.5t_1e_5_pp1
# model_name=quiet_cockatoo_8b_3.5t_2e_5_4000_pp1
# model_name=original_8b_3.5t
num_ctxs=5

# ## single-turn qa (batch-1)
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh nq 8b greedy test  0 200 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh sandia 8b greedy test 0 1000 $num_ctxs $model_name true

## single-turn qa (batch-2)
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh BioASQ 8b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh DuoRC_ParaphraseRC 8b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh boolq 8b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh msmarco 8b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh multirc 8b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh race 8b greedy test 0 1000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh TextbookQA 8b greedy test 0 1000 $num_ctxs $model_name true

# ## multi-turn qa
# # doc2dial 3939 samples
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh doc2dial 8b greedy test 0 1000 $num_ctxs $model_name true
# # quac 7354 samples
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh quac 8b greedy test 0 1000 $num_ctxs $model_name true
# # qrecc 2805 samples
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh qrecc 8b greedy test 0 1000 $num_ctxs $model_name true
# # sharc 10000 samples
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_zeroshot_sft_conv.sh sharc 8b greedy test 0 1000 $num_ctxs $model_name true
