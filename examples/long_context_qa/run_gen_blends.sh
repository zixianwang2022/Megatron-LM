# model_name=qa_blendv12_gpt_1e-8_conv_quiet_cockatoo_pp1_fixed_newsqa_same_format_ctx1_43b_64_3e-7
# model_name=qa_blendv13_gpt_1e-8_conv_quiet_cockatoo_pp1_same_format_ctx1_43b_64_3e-7
# model_name=qa_blendv12_gpt_1e-8_conv_quiet_cockatoo_pp1_fixed_doc2dial_same_format_ctx1_43b_64_3e-7
# model_name=qa_blendv12_gpt_1e-8_conv_quiet_cockatoo_pp1_fixed_doc2dial_fixed_getbatch_same_format_ctx1_43b_64_3e-7
# step=4500
num_ctxs=10
model_name=gpt3-8b-multi-1.1t-gtc-base
step=1417624
model_name=gpt3-8b-multi-1.1t-gtc-itp-16k-lr1e-5
step=1200
# model_name=gpt3-8b-multi-1.1t-gtc-4xbsz
# step=1200
bash examples/long_context_qa/generate_long_sequence_conv.sh nq 8b greedy test  0 200 $step $num_ctxs $model_name true
bash examples/long_context_qa/generate_long_sequence_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 $step $num_ctxs $model_name true
bash examples/long_context_qa/generate_long_sequence_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 $step $num_ctxs $model_name true
bash examples/long_context_qa/generate_long_sequence_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 8b greedy test 0 250 $step $num_ctxs $model_name true
bash examples/long_context_qa/generate_long_sequence_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 $step $num_ctxs $model_name true
bash examples/long_context_qa/generate_long_sequence_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 $step $num_ctxs $model_name true
bash examples/long_context_qa/generate_long_sequence_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 $step $num_ctxs $model_name true
bash examples/long_context_qa/generate_long_sequence_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 $step $num_ctxs $model_name true

