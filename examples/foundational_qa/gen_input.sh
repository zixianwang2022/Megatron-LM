
sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/$TASK/${split}.json"
DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/$TASK"
FEWSHOT_INPUT_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa"

if [[ $TASK == "nq" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/${split}.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/NQ/fewshot_samples.json"
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ"
fi

if [[ $TASK == "tqa" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA/${split}.json"
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA"
fi

if [[ $TASK == att* ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/data/att/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/data/att/$TASK/${split}.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/att/fewshot_samples.json"
fi

if [[ $TASK == nv_benefits* ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/${split}.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/nv_benefits/fewshot_samples.json"
fi

if [[ $TASK == Iternal* ]]; then
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/iternal/fewshot_samples.json"
fi

if [[ $TASK == NVIT* ]]; then
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/nvit/fewshot_samples.json"
fi

if [[ $TASK == ford* ]]; then
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/ford/fewshot_samples.json"
fi

if [[ $TASK == landrover* ]]; then
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/landrover/fewshot_samples.json"
fi

if [[ $TASK == "doc2dial" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/doc2dial/fewshot_samples.json"
fi

if [[ $TASK == "quac" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/quac/fewshot_samples.json"
fi

if [[ $TASK == "qrecc" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/qrecc/fewshot_samples.json"
fi

if [[ $TASK == "sharc" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/sharc/fewshot_samples.json"
fi
