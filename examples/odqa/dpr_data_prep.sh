#!/bin/bash


### TQA Train Data
SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-train-dpr-hn-all-16-ideal-with-answer-validation-full.json

# SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-train-dpr-hn-16-ideal-new.json
# SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-train-dpr-hn-all-16-ideal-with-goldenanswer-validation-full.json
# SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-train-dpr-hn-24-full.json
GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/train.json #\ (e.g., /testseen_knowledge_reference.txt)
MODEL_GEN_PATH_LIST=""
for i in `seq 1 16`
        do
        MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt,"
        done
echo $MODEL_GEN_PATH_LIST

CONTEXT_GEN_PATH_LIST=""
for i in `seq 1 16`
        do
        CONTEXT_GEN_PATH_LIST="${CONTEXT_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/train/generated_context_k10_357m_multisetdpr_queryctx_p0.9_all_rnd${i}.txt,"
        done
echo $CONTEXT_GEN_PATH_LIST

### TQA Test Data
# SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-test-dpr-hn-all-16-ideal-with-answer-validation-full.json

# # SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-test-dpr-hn-all-8-ideal-new.json
# # SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-test-dpr-hn-all-16-new-full.json
# # SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-test-dpr-hn-all-16-ideal-with-goldenanswer-validation-full.json
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/TQA/test.json #\ (e.g., /testseen_knowledge_reference.txt)
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 16`
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


### TQA Dev Data
# SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/TQA/tqa-dev-dpr-hn.json
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/open_domain_data/TQA/dev.json #\ (e.g., /testseen_knowledge_reference.txt)
# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/outputs/dasu/prompting/predicted/TQA/357m/dev/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd${i}_withprob.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

# CONTEXT_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         CONTEXT_GEN_PATH_LIST="${CONTEXT_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/outputs/dasu/prompting/predicted/TQA/GenCTX/357m/dev/generated_context_k10_357m_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $CONTEXT_GEN_PATH_LIST

## NQ Train Data
# SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/NQ/nq-train-dpr-hn-24.json
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/train.json 
# MODEL_GEN_PATH_LIST=""


# for i in `seq 1 24`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

# CONTEXT_GEN_PATH_LIST=""
# for i in `seq 1 24`
#         do
#         CONTEXT_GEN_PATH_LIST="${CONTEXT_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}.txt,"
#         done
# echo $CONTEXT_GEN_PATH_LIST


#### NQ dev data

# SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/NQ/nq-dev-dpr-hn.json
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/open_domain_data/NQ/dev.json 
# MODEL_GEN_PATH_LIST=""


# for i in `seq 1 16`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/outputs/dasu/prompting/predicted/NQ/dev/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd${i}_withprob.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

# CONTEXT_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         CONTEXT_GEN_PATH_LIST="${CONTEXT_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/outputs/dasu/prompting/predicted/NQ/GenCTX/dev/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $CONTEXT_GEN_PATH_LIST

### NQ test data

# SAVE_DPR_DATA_PATH=/gpfs/fs1/projects/gpu_adlr/outputs/dasu/dpr/prompting/data/retriever/NQ/nq-test-dpr-hn-16-full.json
# GROUND_TRUTH_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/dasu/open_domain_data/NQ/test.json 

# MODEL_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         MODEL_GEN_PATH_LIST="${MODEL_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $MODEL_GEN_PATH_LIST

# CONTEXT_GEN_PATH_LIST=""
# for i in `seq 1 16`
#         do
#         CONTEXT_GEN_PATH_LIST="${CONTEXT_GEN_PATH_LIST}/gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_rnd${i}.txt,"
#         done
# echo $CONTEXT_GEN_PATH_LIST



python3 ./tasks/odqa/dpr_data_prep.py \
        --guess-file ${MODEL_GEN_PATH_LIST} \
        --answer-file ${GROUND_TRUTH_PATH} \
        --save-context-path ${CONTEXT_GEN_PATH_LIST} \
        --save-dpr-file-path ${SAVE_DPR_DATA_PATH} \
        --n-neg-samples 15 \
        # --save-similarity-file-path ${SIMILARITY_FILE} \
        # --compare-file ${COMPARE_FILE} \

