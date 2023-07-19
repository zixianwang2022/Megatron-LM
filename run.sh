# 43B GPT
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 0    400    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 400  400    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 800  400    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 1200 400    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 1600 400    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2000 400    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2400 400    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2800 400    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 3200 800    472541 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-multi-1.1t-gtc/tp8pp1
#
#cat generate_43b_test_greedy_0_400_472541.txt \
#     generate_43b_test_greedy_400_400_472541.txt \
#     generate_43b_test_greedy_800_400_472541.txt \
#    generate_43b_test_greedy_1200_400_472541.txt \
#    generate_43b_test_greedy_1600_400_472541.txt \
#    generate_43b_test_greedy_2000_400_472541.txt \
#    generate_43b_test_greedy_2400_400_472541.txt \
#    generate_43b_test_greedy_2800_400_472541.txt \
#    generate_43b_test_greedy_3200_800_472541.txt > generate_43b_test_greedy_0_400_472541.concat.txt
#
#
##bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 0    400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 400  400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 800  400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 1200 400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 1600 400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2000 400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2400 400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2800 400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 3200 800    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1
#
cat  generate_nq_43b_test_greedy_0_400_32552.txt \
     generate_nq_43b_test_greedy_400_400_32552.txt \
     generate_nq_43b_test_greedy_800_400_32552.txt \
    generate_nq_43b_test_greedy_1200_400_32552.txt \
    generate_nq_43b_test_greedy_1600_400_32552.txt \
    generate_nq_43b_test_greedy_2000_400_32552.txt \
    generate_nq_43b_test_greedy_2400_400_32552.txt \
    generate_nq_43b_test_greedy_2800_400_32552.txt \
    generate_nq_43b_test_greedy_3200_800_32552.txt > generate_nq_43b_test_greedy_0_400_32552.concat.txt



## 43B Retro
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 0      400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 400    400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 800    400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 1200   400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 1600   400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2000   400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2400   400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 2800   400    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
bash examples/qa/retro_generate_multijob_ckpt_step_same_format_short.sh nq 43b greedy test 32 1e-6 3200   800    32552 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2

#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 0      400    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 400    400    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 800    400    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1200   400    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1600   400    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2000   400    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2400   400    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2800   400    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 3200   800    31000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2

#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 0      400    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 400    400    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 800    400    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1200   400    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1600   400    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2000   400    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2400   400    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2800   400    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 3200   800    32000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 0      400    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 400    400    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 800    400    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1200   400    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1600   400    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2000   400    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2400   400    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2800   400    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 3200   800    30000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 0      400    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 400    400    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 800    400    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1200   400    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1600   400    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2000   400    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2400   400    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2800   400    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 3200   800    29000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 0      400    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 400    400    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 800    400    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1200   400    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 1600   400    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2000   400    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2400   400    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 2800   400    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 43b greedy test 32 1e-6 3200   800    28000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2



#
#cat    generate_43b_test_greedy_0_400_27000.txt \
#     generate_43b_test_greedy_400_400_27000.txt \
#     generate_43b_test_greedy_800_400_27000.txt \
#    generate_43b_test_greedy_1200_400_27000.txt \
#    generate_43b_test_greedy_1600_400_27000.txt \
#    generate_43b_test_greedy_2000_400_27000.txt \
#    generate_43b_test_greedy_2400_400_27000.txt \
#    generate_43b_test_greedy_2800_400_27000.txt \
#    generate_43b_test_greedy_3200_800_27000.txt > generate_43b_test_greedy_0_400_27000.concat.txt

#cat    generate_43b_test_greedy_0_400_28000.txt \
#     generate_43b_test_greedy_400_400_28000.txt \
#     generate_43b_test_greedy_800_400_28000.txt \
#    generate_43b_test_greedy_1200_400_28000.txt \
#    generate_43b_test_greedy_1600_400_28000.txt \
#    generate_43b_test_greedy_2000_400_28000.txt \
#    generate_43b_test_greedy_2400_400_28000.txt \
#    generate_43b_test_greedy_2800_400_28000.txt \
#    generate_43b_test_greedy_3200_800_28000.txt > generate_43b_test_greedy_0_400_28000.concat.txt
#
#
#cat    generate_43b_test_greedy_0_400_29000.txt \
#     generate_43b_test_greedy_400_400_29000.txt \
#     generate_43b_test_greedy_800_400_29000.txt \
#    generate_43b_test_greedy_1200_400_29000.txt \
#    generate_43b_test_greedy_1600_400_29000.txt \
#    generate_43b_test_greedy_2000_400_29000.txt \
#    generate_43b_test_greedy_2400_400_29000.txt \
#    generate_43b_test_greedy_2800_400_29000.txt \
#    generate_43b_test_greedy_3200_800_29000.txt > generate_43b_test_greedy_0_400_29000.concat.txt
#
#
#cat    generate_43b_test_greedy_0_400_30000.txt \
#     generate_43b_test_greedy_400_400_30000.txt \
#     generate_43b_test_greedy_800_400_30000.txt \
#    generate_43b_test_greedy_1200_400_30000.txt \
#    generate_43b_test_greedy_1600_400_30000.txt \
#    generate_43b_test_greedy_2000_400_30000.txt \
#    generate_43b_test_greedy_2400_400_30000.txt \
#    generate_43b_test_greedy_2800_400_30000.txt \
#    generate_43b_test_greedy_3200_800_30000.txt > generate_43b_test_greedy_0_400_30000.concat.txt
#
#
#cat    generate_43b_test_greedy_0_400_31000.txt \
#     generate_43b_test_greedy_400_400_31000.txt \
#     generate_43b_test_greedy_800_400_31000.txt \
#    generate_43b_test_greedy_1200_400_31000.txt \
#    generate_43b_test_greedy_1600_400_31000.txt \
#    generate_43b_test_greedy_2000_400_31000.txt \
#    generate_43b_test_greedy_2400_400_31000.txt \
#    generate_43b_test_greedy_2800_400_31000.txt \
#    generate_43b_test_greedy_3200_800_31000.txt > generate_43b_test_greedy_0_400_31000.concat.txt
#
#cat    generate_43b_test_greedy_0_400_32000.txt \
#   generate_43b_test_greedy_400_400_32000.txt \
#   generate_43b_test_greedy_800_400_32000.txt \
#  generate_43b_test_greedy_1200_400_32000.txt \
#  generate_43b_test_greedy_1600_400_32000.txt \
#  generate_43b_test_greedy_2000_400_32000.txt \
#  generate_43b_test_greedy_2400_400_32000.txt \
#  generate_43b_test_greedy_2800_400_32000.txt \
#  generate_43b_test_greedy_3200_800_32000.txt > generate_43b_test_greedy_0_400_32000.concat.txt



#cat    generate_43b_test_greedy_0_400_32552.txt \
#     generate_43b_test_greedy_400_400_32552.txt \
#     generate_43b_test_greedy_800_400_32552.txt \
#    generate_43b_test_greedy_1200_400_32552.txt \
#    generate_43b_test_greedy_1600_400_32552.txt \
#    generate_43b_test_greedy_2000_400_32552.txt \
#    generate_43b_test_greedy_2400_400_32552.txt \
#    generate_43b_test_greedy_2800_400_32552.txt \
#    generate_43b_test_greedy_3200_800_32552.txt > generate_43b_test_greedy_0_400_32552.concat.txt


#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/generate_43b_test_greedy_0_400_27000.concat.txt.period.txt
#3610it [00:00, 43029.63it/s]
#Exact Match: 0.3155;
#done :-)

# 22B GPT
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 0    400    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 400  400    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 800  400    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 1200 400    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 1600 400    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 2000 400    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 2400 400    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 2800 400    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 3200 800    708812 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc
#
#cat    generate_22b_test_greedy_0_400_708812.txt \
#     generate_22b_test_greedy_400_400_708812.txt \
#     generate_22b_test_greedy_800_400_708812.txt \
#    generate_22b_test_greedy_1200_400_708812.txt \
#    generate_22b_test_greedy_1600_400_708812.txt \
#    generate_22b_test_greedy_2000_400_708812.txt \
#    generate_22b_test_greedy_2400_400_708812.txt \
#    generate_22b_test_greedy_2800_400_708812.txt \
#    generate_22b_test_greedy_3200_800_708812.txt > generate_22b_test_greedy_0_400_708812.concat.txt
#
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 0    400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 400  400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 800  400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 1200 400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 1600 400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 2000 400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 2400 400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 2800 400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 22b greedy test 32 1e-6 3200 800    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting
#
#cat    generate_22b_test_greedy_0_400_48828.txt \
#     generate_22b_test_greedy_400_400_48828.txt \
#     generate_22b_test_greedy_800_400_48828.txt \
#    generate_22b_test_greedy_1200_400_48828.txt \
#    generate_22b_test_greedy_1600_400_48828.txt \
#    generate_22b_test_greedy_2000_400_48828.txt \
#    generate_22b_test_greedy_2400_400_48828.txt \
#    generate_22b_test_greedy_2800_400_48828.txt \
#    generate_22b_test_greedy_3200_800_48828.txt > generate_22b_test_greedy_0_400_48828.concat.txt
#
## 22B Retro
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 0      400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 400    400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 800    400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 1200   400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 1600   400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 2000   400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 2400   400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 2800   400    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 22b greedy test 32 1e-6 3200   800    48828 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/ 2
#
#cat    generate_22b_test_greedy_0_400_48828.txt \
#     generate_22b_test_greedy_400_400_48828.txt \
#     generate_22b_test_greedy_800_400_48828.txt \
#    generate_22b_test_greedy_1200_400_48828.txt \
#    generate_22b_test_greedy_1600_400_48828.txt \
#    generate_22b_test_greedy_2000_400_48828.txt \
#    generate_22b_test_greedy_2400_400_48828.txt \
#    generate_22b_test_greedy_2800_400_48828.txt \
#    generate_22b_test_greedy_3200_800_48828.txt > generate_22b_test_greedy_0_400_48828.concat.txt
#
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-multi-1.1t-gtc/generate_22b_test_greedy_0_400_708812.concat.txt.period.txt
#3610it [00:00, 46607.54it/s]
#Exact Match: 0.2922;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-gpt-fitting/generate_22b_test_greedy_0_400_48828.concat.txt.period.txt
#3610it [00:00, 38060.07it/s]
#Exact Match: 0.2934;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-22b-pretraining-retro-fitting-noseqpar/generate_22b_test_greedy_0_400_48828.concat.txt.period.txt
#3610it [00:00, 45890.93it/s]
#Exact Match: 0.3033;
#done :-)

# 8B GPT
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 0    400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 400  400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 800  400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 1200 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 1600 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 2000 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 2400 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 2800 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 3200 800    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc
#
#cat    generate_8b_test_greedy_0_400_1417624.txt \
#     generate_8b_test_greedy_400_400_1417624.txt \
#     generate_8b_test_greedy_800_400_1417624.txt \
#    generate_8b_test_greedy_1200_400_1417624.txt \
#    generate_8b_test_greedy_1600_400_1417624.txt \
#    generate_8b_test_greedy_2000_400_1417624.txt \
#    generate_8b_test_greedy_2400_400_1417624.txt \
#    generate_8b_test_greedy_2800_400_1417624.txt \
#    generate_8b_test_greedy_3200_800_1417624.txt > generate_8b_test_greedy_0_400_1417624.concat.txt
#
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 0    400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 400  400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 800  400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 1200 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 1600 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 2000 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 2400 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 2800 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 8b greedy test 32 1e-6 3200 800    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting
#
#cat      generate_8b_test_greedy_0_400_97656.txt \
#     generate_8b_test_greedy_400_400_1417624.txt \
#     generate_8b_test_greedy_800_400_1417624.txt \
#    generate_8b_test_greedy_1200_400_1417624.txt \
#    generate_8b_test_greedy_1600_400_1417624.txt \
#    generate_8b_test_greedy_2000_400_1417624.txt \
#    generate_8b_test_greedy_2400_400_1417624.txt \
#    generate_8b_test_greedy_2800_400_1417624.txt \
#    generate_8b_test_greedy_3200_800_1417624.txt > generate_8b_test_greedy_0_400_97656.concat.txt
#
#
## 8B Retro
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 0      400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 400    400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 800    400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 1200   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 1600   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 2000   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 2400   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 2800   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 8b greedy test 32 1e-6 3200   800    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/ 2
#
#
#cat      generate_8b_test_greedy_0_400_97656.txt \
#     generate_8b_test_greedy_400_400_97656.txt \
#     generate_8b_test_greedy_800_400_97656.txt \
#    generate_8b_test_greedy_1200_400_97656.txt \
#    generate_8b_test_greedy_1600_400_97656.txt \
#    generate_8b_test_greedy_2000_400_97656.txt \
#    generate_8b_test_greedy_2400_400_97656.txt \
#    generate_8b_test_greedy_2800_400_97656.txt \
#    generate_8b_test_greedy_3200_800_97656.txt > generate_8b_test_greedy_0_400_97656.concat.txt


#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-1.1t-gtc/generate_8b_test_greedy_0_400_1417624.concat.txt.period.txt
#3610it [00:00, 43266.44it/s]
#Exact Match: 0.2443;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting/generate_8b_test_greedy_0_400_97656.concat.txt.period.txt
#3610it [00:00, 46487.77it/s]
#Exact Match: 0.2551;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-noseqpar/generate_8b_test_greedy_0_400_97656.concat.txt.period.txt
#3610it [00:00, 52093.48it/s]
#Exact Match: 0.2593;
#done :-)



# 2B GPT
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 0    400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 400  400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 800  400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 1200 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 1600 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 2000 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 2400 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 2800 400    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 3200 800    1417624 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc
#
#cat generate_2b_test_greedy_0_400_1417624.txt \
#    generate_2b_test_greedy_400_400_1417624.txt \
#    generate_2b_test_greedy_800_400_1417624.txt \
#    generate_2b_test_greedy_1200_400_1417624.txt \
#    generate_2b_test_greedy_1600_400_1417624.txt \
#    generate_2b_test_greedy_2000_400_1417624.txt \
#    generate_2b_test_greedy_2400_400_1417624.txt \
#    generate_2b_test_greedy_2800_400_1417624.txt \
#    generate_2b_test_greedy_3200_800_1417624.txt > generate_2b_test_greedy_0_400_1417624.concat.txt
#
#
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 0    400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 400  400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 800  400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 1200 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 1600 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 2000 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 2400 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 2800 400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 2b greedy test 32 1e-6 3200 800    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting
#
#cat    generate_2b_test_greedy_0_400_97656.txt \
#     generate_2b_test_greedy_400_400_97656.txt \
#     generate_2b_test_greedy_800_400_97656.txt \
#    generate_2b_test_greedy_1200_400_97656.txt \
#    generate_2b_test_greedy_1600_400_97656.txt \
#    generate_2b_test_greedy_2000_400_97656.txt \
#    generate_2b_test_greedy_2400_400_97656.txt \
#    generate_2b_test_greedy_2800_400_97656.txt \
#    generate_2b_test_greedy_3200_800_97656.txt > generate_2b_test_greedy_0_400_97656.concat.txt
#
## 2B Retro
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 0      400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 400    400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 800    400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 1200   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 1600   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 2000   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 2400   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 2800   400    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 2b greedy test 32 1e-6 3200   800    97656 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting 2
##
##
#cat generate_2b_test_greedy_0_400_97656.txt \
#    generate_2b_test_greedy_400_400_97656.txt \
#    generate_2b_test_greedy_800_400_97656.txt \
#    generate_2b_test_greedy_1200_400_97656.txt \
#    generate_2b_test_greedy_1600_400_97656.txt \
#    generate_2b_test_greedy_2000_400_97656.txt \
#    generate_2b_test_greedy_2400_400_97656.txt \
#    generate_2b_test_greedy_2800_400_97656.txt \
#    generate_2b_test_greedy_3200_800_97656.txt > generate_2b_test_greedy_0_400_97656.concat.txt

#python truncate_qa_output.py

#/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting/generate_2b_test_greedy_0_400_97656.concat.txt.period.txt

#python tasks/retro_qa/evaluate.py

#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting/generate_2b_test_greedy_0_400_97656.concat.txt.period.txt
#3610it [00:00, 58307.39it/s]
#Exact Match: 0.0427;
#done :-)

#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-multi-1.1t-gtc/generate_2b_test_greedy_0_400_1417624.concat.txt.period.txt
#3610it [00:00, 19397.50it/s]
#Exact Match: 0.0740;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-gpt-fitting/generate_2b_test_greedy_0_400_97656.concat.txt.period.txt
#3610it [00:00, 18296.93it/s]
#Exact Match: 0.0956;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting/generate_2b_test_greedy_0_400_97656.concat.txt.period.txt
#3610it [00:00, 41263.28it/s]
#Exact Match: 0.1551;
#done :-)


# 800M GPT
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 0    400    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 400  400    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 800  400    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 1200 400    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 1600 400    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 2000 400    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 2400 400    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 2800 400    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 3200 800    2835248 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr
##
#cat generate_843m_test_greedy_0_400_2835248.txt \
#    generate_843m_test_greedy_400_400_2835248.txt \
#    generate_843m_test_greedy_800_400_2835248.txt \
#    generate_843m_test_greedy_1200_400_2835248.txt \
#    generate_843m_test_greedy_1600_400_2835248.txt \
#    generate_843m_test_greedy_2000_400_2835248.txt \
#    generate_843m_test_greedy_2400_400_2835248.txt \
#    generate_843m_test_greedy_2800_400_2835248.txt \
#    generate_843m_test_greedy_3200_800_2835248.txt > generate_843m_test_greedy_0_400_2835248.concat.txt
##
##
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 0    400    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 400  400    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 800  400    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 1200 400    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 1600 400    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 2000 400    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 2400 400    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 2800 400    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
#bash examples/qa/generate_multijob_ckpt_step_same_format_short.sh nq 843m greedy test 32 1e-6 3200 800    194000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting
##
#cat    generate_843m_test_greedy_0_400_194000.txt \
#     generate_843m_test_greedy_400_400_194000.txt \
#     generate_843m_test_greedy_800_400_194000.txt \
#    generate_843m_test_greedy_1200_400_194000.txt \
#    generate_843m_test_greedy_1600_400_194000.txt \
#    generate_843m_test_greedy_2000_400_194000.txt \
#    generate_843m_test_greedy_2400_400_194000.txt \
#    generate_843m_test_greedy_2800_400_194000.txt \
#    generate_843m_test_greedy_3200_800_194000.txt > generate_843m_test_greedy_0_400_194000.concat.txt
#
#python truncate_qa_output.py
#
#python tasks/retro_qa/evaluate.py


#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr/generate_843m_test_greedy_0_400_2835248.concat.txt.period.txt
#3610it [00:00, 35121.90it/s]
#Exact Match: 0.0158;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting/generate_843m_test_greedy_0_400_194000.concat.txt.period.txt
#3610it [00:00, 33505.20it/s]
#Exact Match: 0.0075;
#done :-)

# 800M Retro
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 0      400    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 400    400    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 800    400    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 1200   400    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 1600   400    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 2000   400    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 2400   400    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 2800   400    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
#bash examples/qa/retro_generate_multijob_ckpt_step_same_format.sh nq 843m greedy test 32 1e-6 3200   800    195312 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting 2
##
##
#cat generate_843m_test_greedy_0_400_195312.txt \
#     generate_843m_test_greedy_400_400_195312.txt \
#     generate_843m_test_greedy_800_400_195312.txt \
#    generate_843m_test_greedy_1200_400_195312.txt \
#    generate_843m_test_greedy_1600_400_195312.txt \
#    generate_843m_test_greedy_2000_400_195312.txt \
#    generate_843m_test_greedy_2400_400_195312.txt \
#    generate_843m_test_greedy_2800_400_195312.txt \
#    generate_843m_test_greedy_3200_800_195312.txt > generate_843m_test_greedy_0_400_195312.concat.txt

#python truncate_qa_output.py
#
#python tasks/retro_qa/evaluate.py
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting/generate_843m_test_greedy_0_400_195312.concat.txt.period.txt
#3610it [00:00, 44255.74it/s]
#Exact Match: 0.0130;
#done :-)


#python tasks/retro_qa/evaluate.py
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-843m-multi-1.1t-gtc-llr/generate_843m_test_greedy_0_400_2835248.concat.txt.period.txt
#3610it [00:00, 24607.90it/s]
#Exact Match: 0.0751;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-gpt-fitting/generate_843m_test_greedy_0_400_194000.concat.txt.period.txt
#3610it [00:00, 17083.72it/s]
#Exact Match: 0.0377;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting/generate_843m_test_greedy_0_400_195312.concat.txt.period.txt
#3610it [00:00, 43680.71it/s]
#Exact Match: 0.0593;
#done :-)


#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro/gpt3-1.3b-pretraining-retro-K-2/generate_nq_1.3b_test_greedy_0_400_2_375000_1_short_format.concat.txt.period.txt
#3610it [00:00, 56164.06it/s]
#Exact Match: 0.1795;
#done :-)
#reading /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/gpt3/gpt3-1.3b/generate_nq_1.3b_test_greedy_0_400_389532_short_format_1.concat.txt.period.txt
#3610it [00:00, 43622.07it/s]
#Exact Match: 0.1504;
#done :-)

#


# for killing
#START=3918710
#END=3918775
#for ((i=START;i<=END;i++)); do
#    echo $i
#    kill_job $i
#done