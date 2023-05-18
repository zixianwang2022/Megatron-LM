ckpt_steps=("30" "60" "90" "120" "150")
# retrievers=("dragon_retriever_msmarcominilm_reranker" "tasb_msmarcominilm")
retrievers=("dragon_retriever_msmarcominilm_reranker")
retrievers=("ft_dragon_retriever_ftmsmarcominilm_reranker")
retrievers=("tasb_msmarcominilm")
tops=("1" "5")
# tops=("1" "5" "10" "15")
chunk_size=300
for retriever in "${retrievers[@]}"
do
  for top in "${tops[@]}"
  do
    for ckpt_step in "${ckpt_steps[@]}"
    do
    cmd="python3 evaluate_att.py --datapath /lustre/fsw/portfolios/adlr/users/pengx/data/att/att_${retriever}_chunkbysents${chunk_size}_retrieved/test.json \
    --gen_test_file /lustre/fsw/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/att_${retriever}_chunkbysents${chunk_size}_retrieved_gpt_same_format_ctx${top}_43b_8_1e-6/generate_43b_test_greedy_0_200_${ckpt_step}.txt"
    echo $cmd
    $cmd
    done
  done
done

# python3 evaluate_att.py --datapath /lustre/fsw/portfolios/adlr/users/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json --gen_test_file /lustre/fsw/portfolios/adlr/users/pengx/projects/43b_gpt_QA/checkpoints/applications/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_gpt_same_format_ctx5_43b_8_1e-6_shuffle_topn/generate_43b_test_greedy_0_200_30.txt
