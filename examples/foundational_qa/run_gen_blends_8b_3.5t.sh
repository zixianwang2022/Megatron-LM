
# model_name=multiturn_qa_blendv2_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7
# model_name=multiturn_qa_blendv1_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7

# model_name=multiturn_qa_blend_commercial_v5_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v13_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v14_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7
# model_name=multiturn_qa_blend_commercial_v15_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7
model_name=multiturn_qa_blend_commercial_gpt3_quiet_cockatoo_8b_3.5t_5e_6_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7

num_ctxs=5

# ## run on doc2dial_gold
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial_gold 8b greedy test  0 4000 4000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial_gold_v2 8b greedy test  0 4000 4000 $num_ctxs $model_name true

## run on the full set of NQ
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 4000 4000 $num_ctxs $model_name true

## single-turn qa (batch-1)
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 200 4000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 4000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 4000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 4000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 4000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 4000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 4000 $num_ctxs $model_name true
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh sandia 8b greedy test 0 1000 4000 $num_ctxs $model_name true

# ## single-turn-qa (batch-2)
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh BioASQ 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh DuoRC_ParaphraseRC 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh boolq 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh msmarco 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh multirc 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh race 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh TextbookQA 8b greedy test 0 1000 4000 $num_ctxs $model_name true

## multi-turn qa
# doc2dial 3939 samples
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# quac 7354 samples
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh quac 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# qrecc 2805 samples
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh qrecc 8b greedy test 0 1000 4000 $num_ctxs $model_name true
# sharc 10000 samples
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh sharc 8b greedy test 0 1000 4000 $num_ctxs $model_name true
