# DATA_HOME="/lustre/fsw/portfolios/llmservice/users/chankyul/s3_data/" #"/lustre/fsw/portfolios/adlr/users/pengx/projects/retro/"
DATA_HOME="/lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/"
data_folder="$DATA_HOME"

# QA_HOME="/lustre/fsw/portfolios/adlr/users/pengx/projects/sft_43b_qa"
# MOUNTS="/lustre/fsw/portfolios"
# PARTITION="batch_block1,batch_block2"
# QA_HOME="/lustre/fsw/adlr/adlr-nlp/pengx/sft_43b_qa/"
QA_HOME="/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa"
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

PRETRAINED_CHECKPOINT="/lustre/fsw/adlr/adlr-nlp/mpatwary/checkpoints/gpt3/share-checkpoints/${NAME}"
unbiased_cuckoo="/lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/chat_oasst_43B_tp8_pp1"
quiet_cockatoo="/lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/megatron_sft_quiet_cockatoo_tp8_pp1/"
malachite_sawfly="/lustre/fsw/adlr/adlr-nlp/pengx/megatron-lm/tools/megatron_ckpts/malachite-sawfly/"
neat_spoonbill="/lustre/fsw/adlr/adlr-nlp/pengx/megatron-lm/tools/megatron_ckpts/neat-spoonbill/"
vehement_coyote="/lustre/fsw/adlr/adlr-nlp/pengx/megatron-lm/tools/megatron_ckpts/vehement-coyote/"
wine_jackdaw="/lustre/fsw/adlr/adlr-nlp/pengx/megatron-lm/tools/megatron_ckpts/wine-jackdaw/"
TOKENIZER_MODEL="/lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
