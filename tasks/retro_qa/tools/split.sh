# for retriever in dd
retrievers=("dragon_retriever_msmarcominilm_reranker" "tasb_msmarcominilm")
chunk_sizes=("150" "300")
splits=("train" "test")
for retriever in "${retrievers[@]}"
do
  for chunk_size in "${chunk_sizes[@]}"
  do
    for split in "${splits[@]}"
    do
    #  retriever=dragon_retriever_msmarcominilm_reranker
    #  chunk_size=150
    #  split=test
    full_retrieved_data_file=/lustre/fsw/portfolios/adlr/users/pengx/data/att/experiments/att_qa_${retriever}_chunkbysents${chunk_size}_retriever_top10.json
    index_file=/lustre/fsw/portfolios/adlr/users/pengx/data/att/raw/att_qa_${split}.json
    output_file=/lustre/fsw/portfolios/adlr/users/pengx/data/att/att_${retriever}_chunkbysents${chunk_size}_retrieved/${split}.json
    mkdir -p /lustre/fsw/portfolios/adlr/users/pengx/data/att/att_${retriever}_chunkbysents${chunk_size}_retrieved/
    # python3 construct_train_test_with_retrieved.py --full_retrieved_data_file ${full_retrieved_data_file} --index_file $index_file --output_file $output_file
    echo $output_file
    done
  done
done
