src_dir="/lustre/fsw/adlr/adlr-nlp/pengx/sft_70b_llama2_qa/megatron-lm"
output_base_dir="/lustre/fsw/adlr/adlr-nlp/pengx/long_context_llm/megatron-lm/eval_data"
ADLR_NLP_SHARING="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing"
prefix="pg19_test"
filename="pg19.json"
cpus_per_node=1
python -u ${src_dir}/tools/preprocess_data.py \
      --input ${output_base_dir}/json/${prefix}/${filename} \
      --output-prefix ${output_base_dir}/bin-idx/${prefix}/${prefix}.llama2  \
      --dataset-impl mmap \
      --tokenizer-type Llama2Tokenizer \
      --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2/tokenizer.model \
      --workers ${cpus_per_node} \
      --log-interval 10 \
      --partitions 1 \
      --append-eod
