#DATA_HOME="../../inform-retriever-msmarco"
DATA_HOME="/lustre/fsw/adlr/adlr-nlp/pengx/retro"

sample_input_file="${DATA_HOME}/data/$TASK/${split}.json"
DATA_FOLDER="${DATA_HOME}/data/$TASK"

if [[ $TASK == "nq" ]]; then
    sample_input_file="${DATA_HOME}/data/NQ/${split}.json"
    DATA_FOLDER="${DATA_HOME}/data/NQ/$TASK"
fi

if [[ $TASK == "tqa" ]]; then
    sample_input_file="${DATA_HOME}/data/TQA/${split}.json"
    DATA_FOLDER="${DATA_HOME}/data/TQA/$TASK"
fi

if [[ $TASK == Iternal* ]]; then
    sample_input_file="${DATA_HOME}/data/Iternal/$TASK/${split}.json"
    DATA_FOLDER="${DATA_HOME}/data/Iternal/$TASK"
fi

if [[ $TASK == carmanual* ]]; then
    sample_input_file="${DATA_HOME}/data/carmanual/$TASK/${split}.json"
    DATA_FOLDER="${DATA_HOME}/data/carmanual/$TASK"
fi

if [[ $TASK == NVIT* ]]; then
    sample_input_file="${DATA_HOME}/data/NVIT/$TASK/${split}.json"
    DATA_FOLDER="${DATA_HOME}/data/NVIT/$TASK"
fi

if [[ $TASK == TCS* ]]; then
    sample_input_file="${DATA_HOME}/data/TCS/$TASK/${split}.json"
    DATA_FOLDER="${DATA_HOME}/data/TCS/$TASK"
fi

if [[ $TASK == att* ]]; then
    sample_input_file="${DATA_HOME}/data/att/$TASK/${split}.json"
    DATA_FOLDER="${DATA_HOME}/data/att/$TASK"
fi

QA_HOME="/lustre/fsw/adlr/adlr-nlp/boxinw/nvllm-qa"
MOUNTS="/lustre/fsw/adlr/adlr-nlp/"
PARTITION="luna"
LAUNCH="${ADLR_UTILS}/mp_launch"


NAME="gpt3-${model_size}-multi-1.1t-gtc"
if [[ ${model_size} == "843m" ]]; then
    NAME="gpt3-843m-multi-1.1t-gtc-llr"
fi

if [[ ${model_size} == "43b" ]]; then
    NAME="gpt3-43b-multi-1.1t-gtc/tp8pp4"
fi

#PRETRAINED_CHECKPOINT="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/${NAME}"
#if [[ ${model_card} == *-pp1-v1 ]]; then
#    ##v1
#    PRETRAINED_CHECKPOINT="/lustre/fs1/portfolios/adlr/users/jkamalu/checkpoint-converter-nemo-to-megatron/checkpoints/mega/megatron_converted_43b_sft_deployed_tp8_pp1-reconverted"
#fi
#if [[ ${model_card} == *-pp1-v2 ]]; then
#    ##v2
#    PRETRAINED_CHECKPOINT="/lustre/fsw/portfolios/llmservice/users/chankyul/megatron_qa_2/mega/megatron_converted_chat_oasst_43B_tp8"
#fi
TOKENIZER_MODEL="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
