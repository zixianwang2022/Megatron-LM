# DATA_HOME="/lustre/fsw/portfolios/llmservice/users/chankyul/s3_data/" #"/lustre/fsw/portfolios/adlr/users/pengx/projects/retro/"
# DATA_HOME="/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/"
DATA_HOME="/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/"
data_folder="$DATA_HOME"

QA_HOME="/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2"
MOUNTS="/lustre/fsw/adlr/adlr-nlp/"
PARTITION="luna"
LAUNCH="$ADLR_UTILS/mp_launch"


NAME="gpt3-${model_size}-multi-1.1t-gtc"
if [[ ${model_size} == "843m" ]]; then
    NAME="gpt3-843m-multi-1.1t-gtc-llr"
fi

if [[ ${model_size} == "43b" ]]; then
    NAME="gpt3-43b-multi-1.1t-gtc/tp8pp4"
fi

# PRETRAINED_CHECKPOINT="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/${NAME}"
# TOKENIZER_MODEL="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"

# PRETRAINED_CHECKPOINT="/lustre/fsw/adlr/adlr-nlp/mpatwary/checkpoints/gpt3/share-checkpoints/${NAME}"

unbiased_cuckoo="/lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/chat_oasst_43B_tp8_pp1"
quiet_cockatoo="/lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/megatron_sft_quiet_cockatoo_tp8_pp1/"
malachite_sawfly="/lustre/fsw/adlr/adlr-nlp/pengx/megatron-lm/tools/megatron_ckpts/malachite-sawfly/"
neat_spoonbill="/lustre/fsw/adlr/adlr-nlp/pengx/megatron-lm/tools/megatron_ckpts/neat-spoonbill/"
vehement_coyote="/lustre/fsw/adlr/adlr-nlp/pengx/megatron-lm/tools/megatron_ckpts/vehement-coyote/"
wine_jackdaw="/lustre/fsw/adlr/adlr-nlp/pengx/megatron-lm/tools/megatron_ckpts/wine-jackdaw/"

llama2_chat_7b="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2/checkpoints/megatron/chat/7b/"
llama2_text_7b="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2/checkpoints/megatron/text/7b/"
llama2_chat_13b="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2/checkpoints/megatron/chat/13b/"
llama2_text_13b="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2/checkpoints/megatron/text/13b/"
llama2_chat_70b_pp1="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2/checkpoints/megatron/chat/70b/"
llama2_text_70b_pp1="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2/checkpoints/megatron/text/70b/"
llama2_chat_70b="/lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/llama2_megatron_chat_70b_pp4/"
llama2_text_70b="/lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/llama2_megatron_text_70b_pp4/"

llama2_text_70b_with_qc="/lustre/fsw/adlr/adlr-nlp/pengx/sft_70b_llama2_qa/checkpoints/applications/qc_llama2_text_70b_base_70b_128_5e-6/"

TOKENIZER_MODEL="/lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
TOKENIZER_MODEL_LLAMA2="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2/tokenizer.model"
