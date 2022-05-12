#!/bin/bash


CONTEXT_GEN_PATH_LIST=""
# for i in `seq 1 64`
#         do
#         CMD="cat /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_2000_rnd${i}.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_4000_rnd${i}.txt \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_6000_rnd${i}.txt /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_8000_rnd${i}.txt \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_10000_rnd${i}.txt \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_11313_rnd${i}.txt > \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_all_rnd${i}.txt"
#         # echo $CMD
#         eval $CMD
#         lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/generated_context_k10_357m_multisetdpr_queryctx_p0.9_all_rnd${i}.txt | awk {'print $1'})
#         if [ "$lines" -eq "11313" ]; then
#             echo " ${i} successful!"
#         else
#             echo "${i} line not enough!"
#         fi
#         done

# for i in `seq 1 64`
#         do
#         CMD="cat /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_2000_rnd${i}_withprob.txt \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_4000_rnd${i}_withprob.txt \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_6000_rnd${i}_withprob.txt \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_8000_rnd${i}_withprob.txt \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_10000_rnd${i}_withprob.txt \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_11313_rnd${i}_withprob.txt > \
#         /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt"
#         # echo $CMD
#         eval $CMD
#         lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt | awk {'print $1'})
#         if [ "$lines" -eq "11313" ]; then
#             echo " ${i} successful!"
#         else
#             echo "${i} line not enough!"
#         fi
#         done


# for the train data
# for i in `seq 1 16`
#         do
#             CAT_LIST=""
#             for j in {1..8..1}
#                 do
#                     CAT_LIST="${CAT_LIST} /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_${j}_rnd${i}.txt"
#                 done
#             CMD="cat${CAT_LIST} > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}.txt"
#             # echo $CMD
#             eval $CMD

#             lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}.txt | awk {'print $1'})
#             if [ "$lines" -eq "79168" ]; then
#                 echo " ${i} successful!"
#             else
#                 echo "${i} line not enough!"
#             fi
#         done


# for i in {17..17..1}
#         do
#             CAT_LIST=""
#             for j in {1..8..1}
#                 do
#                     CMD="cat /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_${j}_rnd${i}.txt.2 /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_${j}_rnd${i}.txt > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_${j}_rnd${i}.txt.1"
#                     eval $CMD
#                 lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_${j}_rnd${i}.txt.1 | awk {'print $1'})
                
#                 if [ "$lines" -eq "10000" ]; then
#                     echo " ${i} successful!"
#                 else
#                     echo "${i} line not enough!"
#                 fi  
#                 done

#         done



### NQ Train
# for i in {18..24..1}
#         do
#             CAT_LIST=""
#             for j in {1..8..1}
#                 do
#                     CAT_LIST="${CAT_LIST} /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_${j}_rnd${i}_withprob.txt.1"
#                 done
#             CMD="cat${CAT_LIST} > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt"
#             # echo $CMD
#             eval $CMD

#             lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt | awk {'print $1'})
#             if [ "$lines" -eq "79168" ]; then
#                 echo " ${i} successful!"
#             else
#                 echo "${i} line not enough!"
#             fi
#         done

for i in {17..24..1}
        do
            CAT_LIST=""
            for j in {1..8..1}
                do
                    CAT_LIST="${CAT_LIST} /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_${j}_rnd${i}.txt.1"
                done
            CMD="cat${CAT_LIST} > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}.txt"
            # echo $CMD
            eval $CMD

            lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/NQ/GenCTX/train/generated_context_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}.txt | awk {'print $1'})
            if [ "$lines" -eq "79168" ]; then
                echo " ${i} successful!"
            else
                echo "${i} line not enough!"
            fi
        done


### TQA Train
# for i in {17..24..1}
#         do
#             CAT_LIST=""
#             for j in {1..10..1}
#                 do
#                     CAT_LIST="${CAT_LIST} /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_${j}_rnd${i}_withprob.txt"
#                 done
#             CMD="cat${CAT_LIST} > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt"
#             # echo $CMD
#             eval $CMD

#             lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/357m/train/output_answer_generations_k10_357m_gc_multisetdpr_queryctx_p0.9_all_rnd${i}_withprob.txt | awk {'print $1'})
#             if [ "$lines" -eq "78785" ]; then
#                 echo " ${i} successful!"
#             else
#                 echo "${i} line not enough!"
#             fi
#         done

# for i in {17..24..1}
#         do
#             CAT_LIST=""
#             for j in {1..10..1}
#                 do
#                     CAT_LIST="${CAT_LIST} /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/train/generated_context_k10_357m_multisetdpr_queryctx_p0.9_${j}_rnd${i}.txt"
#                 done
#             CMD="cat${CAT_LIST} > /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/train/generated_context_k10_357m_multisetdpr_queryctx_p0.9_all_rnd${i}.txt"
#             # echo $CMD
#             eval $CMD

#             lines=$(eval wc -l /gpfs/fs1/projects/gpu_adlr/datasets/dasu/prompting/predicted/TQA/GenCTX/357m/train/generated_context_k10_357m_multisetdpr_queryctx_p0.9_all_rnd${i}.txt | awk {'print $1'})
#             if [ "$lines" -eq "78785" ]; then
#                 echo " ${i} successful!"
#             else
#                 echo "${i} line not enough!"
#             fi
#         done
