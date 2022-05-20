#!/bin/bash

#########################
# Evaluate the EM scores.
#########################
# pip install transformers==4.19.0 --use-feature=2020-resolver
pip install transformers==4.18.0 --use-feature=2020-resolver


export CUDA_VISIBLE_DEVICES=0

WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9.txt.1,

# NQ 
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_withnewnewGPTPrefix_l10_nogolden_withcontext_norandom_multisetdpr_qc.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_withnewnewGPTPrefix_l10_usegolden_withcontext_norandom_multisetdpr_qqctx.txt 
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_1.3b_gc_357m_multisetdpr_queryctx_p0.9_rnd1.txt # (e.g., /testseen_knowledge_generations.txt)
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/qg/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_fewshot.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/qg/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_wikitop1.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/NQ/openai_ada_ours_np_64.txt # (e.g., /testseen_knowledge_generations.txt)
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/NQ/openai_ada_GPT-3_np_64.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/NQ/openai_ada_Eleuther-AI_np_64.txt
GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json #\ (e.g., /testseen_knowledge_reference.txt)
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9.txt.1
# COMPARE_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/analysi_result/rt_gf.json
# MODEL_GEN_PATH_LIST="/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9.txt.1,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd2.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd3.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd4.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd5.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd6.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd7.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd9.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd9.txt" 
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_rnd1_new.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_rnd1_new_withoutanswerprob.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_rnd1_new_withoutanswerprob.txt

############# this is the 530B baseline
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k64_multisetdpr_queryctx_530b_nocontext.txt
############# this is the 530B gc + 530 ans
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k10_530b_gc_530_ans_rnd1.txt

############# this is the 530B gc + 530 ans, margin

# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k10_530b_gc_530_ans_rnd1.txt

MODEL_GEN_PATH_LIST=""
for i in `seq 1 8`
        do
                MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
        done
echo $MODEL_GEN_PATH_LIST


############# 530gc + 357m
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#                 MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/api/api_output_answer_generations_k10_530b_gc_357m_ans_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

############# 530gc + 1.3B
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 1`
#         do
#                 MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/api_output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# golden + 1.3B ans model
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/output_answer_generations_k10_golden_ctx_multisetdpr_queryctx_p0.9_rnd1.txt

############# top 1+ 1.3B ans model
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_rnd1.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_rnd1_reversed.txt

############# top1 + 357m ans model
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/357m/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_rnd1.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/357m/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_rnd1_reversed.txt

############# The new model 357m gc + 357m ans
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#                 MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/new/output_answer_generations_k10_357m_gc_357m_ans_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# 357m gc + 357m ans beam search
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#                 MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_357m_ans_beam4_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# the retrieve topk + 357m ans model
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#                 MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_top${i}_ctx_357m_ans.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

############# the retrieve topk + 1.3B ans model
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#                 MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/1.3b/output_answer_generations_k10_top${i}_ctx_1.3b_ans.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# original 357m gc + 357m ans

# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#                 MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

# CONTEXT_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         CONTEXT_GEN_PATH_LIST="${CONTEXT_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $CONTEXT_GEN_PATH_LIST

# this is for BertForRanker
# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/NQ_2/predict_ids.txt
# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/NQ_0/predict_ids.txt
# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/357m/similarity_cgens_64_upper.csv

# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/NQ_0/predict_ids_test.txt

# TQA test set
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_357m_withnewnewGPTPrefix_l10_usegolden_withcontext_norandom_multisetdpr_qqctx_traingoodk0.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rmduplicatectx.txt # (e.g., /testseen_knowledge_generations.txt)
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_357m_withnewnewGPTPrefix_l10_nogolden_withcontext_norandom_multisetdpr_qc.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/qg/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_fewshot.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/qg/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_all_wikitop1.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/TQA/openai_ada_ours_np_0.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/TQA/openai_ada_GPT-3_np_0.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/TQA/openai_ada_Eleuther-AI_np_0.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all.top2.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd1.txt
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/test.json #\ (e.g., /testseen_knowledge_reference.txt)
# GEN_CTX_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/generated_context_k10_357m_multisetdpr_queryctx_p0.9_all.txt
# COMPARE_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/analysi_result/rf_gt.json
# MODEL_GEN_PATH_LIST="/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd2.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd3.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd4.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd5.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd6.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd7.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd8.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd9.txt,\
# /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd10.txt" 

# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/api/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_withprob.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd1_withprob.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/api/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_withoutanswerprob.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_all_withprob.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_all_withprob.txt


############# for golden + 1.3B ans model
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/output_answer_generations_k10_golden_ctx_multisetdpr_queryctx_p0.9_rnd1.txt

############# top1 + 1.3B ans model
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_rnd1_reversed.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/output_answer_generations_k10_top1_ctx_multisetdpr_queryctx_p0.9_rnd1.txt

############# top1 + 357m model
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_top1_ctx_rnd1.txt

############# this is for 530B baseline
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/api/output_answer_generations_k64_multisetdpr_queryctx_530b_nocontext.txt

############# this is for 530B + 530B
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/api/output_answer_generations_k10_530b_gc_530_ans_rnd1.txt


############# this is for 530B + 1.3B
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/api_output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_rnd1.txt

############# this is for the train data
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd2_withprob.txt

# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/train.json #\ (e.g., /testseen_knowledge_reference.txt)
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# the new 357m + 357m
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd${i}_new.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

############# the 357m + 357m 
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_357m_ans_beam4_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# for topk retrieval ctx + 357m answer model
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k10_top${i}_ctx_357m_ans.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

############# for topk retrieval ctx + 1.3B answer model
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/1.3b/output_answer_generations_k10_top${i}_ctx_1.3b_ans.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# the original 357m gc + 357m ans model
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


# CONTEXT_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         CONTEXT_GEN_PATH_LIST="${CONTEXT_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_all_rnd${i}.txt,"
#         done
# echo $CONTEXT_GEN_PATH_LIST

# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/TQA_bertforrank_softmax_ideal_multiple_pos_test_new_new/predict_ids_test_test.txt
# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/TQA_bertforrank_softmax_ideal_multiple_pos_test_new_new/predict_results_test_test.csv

# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/TQA_bertforrank0_16/predict_ids_test.txt
# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/TQA/predict_ids.txt

## TQA_bertforrank0_16 model
# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/TQA_bertforrank0_16/predict_ids_test.txt
# the new softmax based results
# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/TQA_bertforrank_softmax_16/predict_ids_test.txt

# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/ranker/TQA_bertforrank_classification/predict_ids_test.txt

# # SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/similarity_cgens_64_all_multiset_ft_ckp39.csv

# SIMILARITY_FILE=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/similarity_cgens_64_all_dpr_ft_ckp39.csv

# TQA dev set
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/output_answer_generations_k64_1.3b_dev.txt # (e.g., /testseen_knowledge_generations.txt)
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/dev.json #\ (e.g., /testseen_knowledge_reference.txt)

# WebQuestions test set
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/qg/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_wikiktop1.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/qg/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_fewshot.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_357m_gc_nq.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/WQ/openai_ada_ours_np_64.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/WQ/openai_babbage_GPT-3_np_64.txt
# # MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/pengx/qa/megatron-lm/tasks/odqa/WQ/openai_ada_Eleuther-AI_np_64.txt
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/WQ/WebQuestions-test.txt #\ (e.g., /testseen_knowledge_reference.txt)
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_357m_gc_tqa.txt

# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/downloads/data/retriever/webq-dev.json
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/retrieval/predictions/dpr/webQuestion/WebQuestions-test.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_357m_gc_dpr.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k0_357m_gc_dpr_random.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/dprwq_vs_tqa.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/dprwq_cgenfromtqa_with_samplesfromdprwq.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/tqa_vs_dprwq.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/retrieval_dprwq_random.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_357m_gc_multisetdpr_new.txt

# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_357m_gc_multisetdpr_new_top1.txt
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generation_retrieve_nogolden_shift1.txt


############# this is 530 gc + 1.3b ans
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/api_output_answer_generation_530b_gc_1.3b_ans_rnd1.txt

# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/api_output_answer_generation_530b_gc_1.3b_ans_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

############# baseline: 530B
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/api/output_answer_generations_k64_multisetdpr_queryctx_530b_nocontext.txt

############# 530 gc + 357m ans model
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/api_output_answer_generation_train_rnd1.txt

# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/api_output_answer_generation_530b_gc_357m_ans_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

############# golden + 1.3B ans model
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/output_answer_generation_golden_1.3b_ans_rnd1.txt


############# topk retrieval + 357m model
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_top${i}_ctx_357m_ans.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# top 1 + 1.3B ans model

# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/output_answer_generation_top1_ctx_1.3b_ans_rnd1_reversed.txt


############# top k marginalization + 1.3B ans model

# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/output_answer_generations_k10_top${i}_ctx_1.3b_ans.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# this is the 1.3B + 1.3B model output
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/1.3b/output_answer_generations_k10_1.3b_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

############# Weq 1.3B + 357m model output
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 1`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/357m/output_answer_generations_k10_1.3b_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


############# the ogrinal 357m + 357m

# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/output_answer_generations_k10_357m_gc_multisetdpr_new_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST


# CONTEXT_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         CONTEXT_GEN_PATH_LIST="${CONTEXT_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/GenCTX/generated_context_k10_357m_gc_multisetdpr_new_rnd${i}.txt,"
#         done
# echo $CONTEXT_GEN_PATH_LIST

############# the 530b gc + 530b ans

# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 8`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/WQ/api/output_answer_generations_k10_530b_gc_multisetdpr_queryctx_p0.9_new_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST




# PIQA
# MODEL_GEN_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/PIQA/output_answer_generations_k0_357m_l50_withgptneostyle_p0.0k1t1.0_new.txt # (e.g., /testseen_knowledge_generations.txt)
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/PIQA/valid-labels.lst #\ (e.g., /testseen_knowledge_reference.txt)


python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/odqa/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 1 \
        --task ODQA-EVAL-EM \
        --guess-file ${MODEL_GEN_PATH_LIST} \
        --answer-file ${GROUND_TRUTH_PATH} \
        # --save-context-path ${CONTEXT_GEN_PATH_LIST} \
        # --save-similarity-file-path ${SIMILARITY_FILE} \
        # --compare-file ${COMPARE_FILE} \


############################################
# Evaluate BLEU, METEOR, and ROUGE-L scores.
############################################

# We follow the nlg-eval (https://github.com/Maluuba/nlg-eval) to 
# evaluate the BLEU, METEOR, and ROUGE-L scores. 

# To evaluate on these metrics, please setup the environments based on 
# the nlg-eval github, and run the corresponding evaluation commands.

# nlg-eval \
#     --hypothesis=<PATH_OF_THE_KNOWLEDGE_GENERATION> \
#     --references=<PATH_OF_THE_GROUND_TRUTH_KNOWLEDGE>
