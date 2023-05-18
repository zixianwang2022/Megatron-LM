for check_step in 1 2 3 4 5
do
    # checkpoint_step=$((79 * check_step))
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh landrover_tasb_retrieved 8b greedy test 8 3e-6 0 250 ${checkpoint_step} 1 
    # checkpoint_step=$((39 * check_step))
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh landrover_tasb_retrieved 8b greedy test 16 3e-6 0 250 ${checkpoint_step} 1 
    # checkpoint_step=$((85 * check_step))
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh benz_clean_plus_landrover_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 16 3e-6 0 250 ${checkpoint_step} 1 

    # checkpoint_step=$((69 * check_step))
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh benz_clean_plus_landrover_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 16 3e-6 0 250 ${checkpoint_step} 1

 
    checkpoint_step=$((170 * check_step))
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 15
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 10
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 5
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 1

    checkpoint_step=$((170 * check_step))
    bash examples/qa/generate_multijob_ckpt_step_same_format_cross.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 15 landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved
    bash examples/qa/generate_multijob_ckpt_step_same_format_cross.sh benz_clean_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 15 landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved
    bash examples/qa/generate_multijob_ckpt_step_same_format_cross.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 10 landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved
    bash examples/qa/generate_multijob_ckpt_step_same_format_cross.sh benz_clean_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 10 landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved
    # bash examples/qa/generate_multijob_ckpt_step_same_format_cross.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 5 landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved
    # bash examples/qa/generate_multijob_ckpt_step_same_format_cross.sh benz_clean_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 8 1e-6 0 250 ${checkpoint_step} 5 landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved

    # checkpoint_step=$((30 * check_step))
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents150_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 10
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents150_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 15
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 10
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 15

    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 5
    # bash examples/qa/generate_multijob_ckpt_step_same_format_shuffle_topn.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 5
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 1
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_tasb_msmarcominilm_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 1
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_tasb_msmarcominilm_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 5

    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents150_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 5
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents150_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 1

    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_ft_dragon_retriever_ftmsmarcominilm_reranker_chunkbysents150_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 5
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_ft_dragon_retriever_ftmsmarcominilm_reranker_chunkbysents150_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 1
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_ft_dragon_retriever_ftmsmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 5
    # bash examples/qa/generate_multijob_ckpt_step_same_format.sh att_ft_dragon_retriever_ftmsmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 8 1e-6 0 200 ${checkpoint_step} 1
 
done
