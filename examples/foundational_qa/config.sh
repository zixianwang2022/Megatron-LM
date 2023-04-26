DATA_HOME="/lustre/fsw/portfolios/llmservice/users/chankyul/s3_data/" #"/lustre/fsw/portfolios/adlr/users/pengx/projects/retro/"
data_folder="$DATA_HOME"

QA_HOME="/lustre/fsw/portfolios/adlr/users/pengx/projects/sft_43b_qa"
MOUNTS="/lustre/fsw/portfolios"
PARTITION="batch_block1,batch_block2"
LAUNCH="$ADLR_UTILS/mp_launch"


NAME="gpt3-${model_size}-multi-1.1t-gtc"
if [[ ${model_size} == "843m" ]]; then
    NAME="gpt3-843m-multi-1.1t-gtc-llr"
fi

if [[ ${model_size} == "43b" ]]; then
    NAME="gpt3-43b-multi-1.1t-gtc/tp8pp4"
fi

PRETRAINED_CHECKPOINT="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/${NAME}"
TOKENIZER_MODEL="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
