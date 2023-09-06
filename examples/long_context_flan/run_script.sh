bash examples/long_context_flan/finetune_normal_lm.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-itp-16k-tian
bash examples/long_context_flan/finetune_normal_lm.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-itp-32k-tian-tp16pp4
bash examples/long_context_flan/finetune_normal_lm.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-base

bash examples/long_context_flan/finetune_normal_lm.sh qc 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-itp-16k-tian
bash examples/long_context_flan/finetune_normal_lm.sh qc 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-itp-32k-tian-tp16pp4

bash examples/long_context_flan/finetune_qc_sandeep.sh gpt3-43b-multi-1.1t-gtc-itp-16k-tian
bash examples/long_context_flan/finetune_qc_sandeep.sh gpt3-43b-multi-1.1t-gtc-itp-32k-tian-tp16pp4
bash examples/long_context_flan/finetune_qc_sandeep.sh gpt3-43b-multi-1.1t-gtc-base
bash examples/long_context_flan/ct_qc_sandeep.sh qc_gpt3-43b-multi-1.1t-gtc-base_43b_128_5e-6_continue_2k

bash examples/long_context_flan/debug.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-itp-16k-tian
bash examples/long_context_flan/debug.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-base
bash examples/long_context_flan/debug2.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-itp-32k-tian-tp16pp4

bash examples/long_context_flan/debug.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-itp-16k-tian
bash examples/long_context_flan/debug.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-base
bash examples/long_context_flan/debug2.sh flanv1 43b  64 3e-7 gpt3-43b-multi-1.1t-gtc-itp-32k-tian-tp16pp4

bash examples/long_context_flan/debug_qc.sh gpt3-43b-multi-1.1t-gtc-itp-16k-tian
bash examples/long_context_flan/debug_qc.sh gpt3-43b-multi-1.1t-gtc-itp-32k-tian-tp16pp4
bash examples/long_context_flan/debug_qc.sh gpt3-43b-multi-1.1t-gtc-base

