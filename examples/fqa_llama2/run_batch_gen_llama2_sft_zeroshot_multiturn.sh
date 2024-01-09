model_name=llama2_chat_7b
# model_name=llama2_text_7b_with_qc

# model_name=llama2_chat_13b
# model_name=llama2_text_13b_with_qc

# model_name=llama2_chat_70b
# model_name=llama2_text_70b_with_qc

num_ctxs=5

# ### multi-turn qa

# ## HybridQA 1490 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh HybridQA 70b greedy test 0 1500 $num_ctxs $model_name true


# ## convfinqav3 1490 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh convfinqav3 70b greedy test 0 1500 $num_ctxs $model_name true
# ## finqav2 1147 samples
# # bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh finqav2 70b greedy test 0 1500 $num_ctxs $model_name true

# ## sqa 3100 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sqa 70b greedy test 0 1500 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sqa 70b greedy test 1500 1600 $num_ctxs $model_name true

# ## doc2dial 3939 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 70b greedy test 0 2000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 70b greedy test 2000 2000 $num_ctxs $model_name true

# ## quac 7354 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 70b greedy test 0 2000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 70b greedy test 2000 2000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 70b greedy test 4000 2000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 70b greedy test 6000 2000 $num_ctxs $model_name true

# ## qrecc 2805 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 70b greedy test 0 2000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 70b greedy test 2000 2000 $num_ctxs $model_name true


# ## coqa 7983 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 70b greedy test 0 2000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 70b greedy test 2000 2000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 70b greedy test 4000 2000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 70b greedy test 6000 2000 $num_ctxs $model_name true


# ## doqa_cooking 1797 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_cooking 70b greedy test 0 2000 $num_ctxs $model_name true
# ## doqa_movies 1884 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_movies 70b greedy test 0 2000 $num_ctxs $model_name true
# ## doqa_travel 1713 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_travel 70b greedy test 0 2000 $num_ctxs $model_name true

# ## topiocqa 2514 samples
# ## num_ctxs = 20
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh topiocqa 70b greedy test 0 1300 20 $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh topiocqa 70b greedy test 1300 1300 20 $model_name true

# ## hybriddial 1111 samples
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh hybriddial 70b greedy test 0 1200 $num_ctxs $model_name true

# ## inscit 502 samples
# ## num_ctxs = 20
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh inscit 70b greedy test 0 550 20 $model_name true


#### for llama2-7b/13b ####

## doc2dial 3939 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 7b greedy test 0 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 7b greedy test 500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 7b greedy test 1000 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 7b greedy test 1500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 7b greedy test 2000 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 7b greedy test 2500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 7b greedy test 3000 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 7b greedy test 3500 500 $num_ctxs $model_name true

## quac 7354 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 0 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 300 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 600 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 900 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 1200 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 1500 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 1800 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 2100 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 2400 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 2700 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 3000 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 3300 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 3600 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 3900 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 4200 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 4500 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 4800 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 5100 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 5400 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 5700 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 6000 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 6300 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 6600 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 6900 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 7b greedy test 7200 300 $num_ctxs $model_name true


## qrecc 2805 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 0 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 300 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 600 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 900 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 1200 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 1500 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 1800 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 2100 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 2400 300 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 7b greedy test 2700 300 $num_ctxs $model_name true


## doqa_cooking 1797 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_cooking 7b greedy test 0 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_cooking 7b greedy test 500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_cooking 7b greedy test 1000 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_cooking 7b greedy test 1500 500 $num_ctxs $model_name true

## doqa_movies 1884 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_movies 7b greedy test 0 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_movies 7b greedy test 500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_movies 7b greedy test 1000 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_movies 7b greedy test 1500 500 $num_ctxs $model_name true

## doqa_travel 1713 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_travel 7b greedy test 0 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_travel 7b greedy test 500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_travel 7b greedy test 1000 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doqa_travel 7b greedy test 1500 500 $num_ctxs $model_name true

## topiocqa 2514 samples
## num_ctxs = 20
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh topiocqa 7b greedy test 0 500 20 $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh topiocqa 7b greedy test 500 500 20 $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh topiocqa 7b greedy test 1000 500 20 $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh topiocqa 7b greedy test 1500 500 20 $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh topiocqa 7b greedy test 2000 600 20 $model_name true

## coqa 7983 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 7b greedy test 0 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 7b greedy test 1000 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 7b greedy test 2000 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 7b greedy test 3000 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 7b greedy test 4000 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 7b greedy test 5000 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 7b greedy test 6000 1000 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh coqa 7b greedy test 7000 1000 $num_ctxs $model_name true

## convfinqav3 1490 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh convfinqav3 7b greedy test 0 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh convfinqav3 7b greedy test 500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh convfinqav3 7b greedy test 1000 500 $num_ctxs $model_name true

## sqa 3100 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sqa 7b greedy test 0 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sqa 7b greedy test 500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sqa 7b greedy test 1000 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sqa 7b greedy test 1500 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sqa 7b greedy test 2000 500 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sqa 7b greedy test 2500 600 $num_ctxs $model_name true

## hybriddial 1111 samples
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh hybriddial 7b greedy test 0 600 $num_ctxs $model_name true
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh hybriddial 7b greedy test 600 600 $num_ctxs $model_name true

## inscit 502 samples
num_ctxs=20
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh inscit 7b greedy test 0 550 20 $model_name true

