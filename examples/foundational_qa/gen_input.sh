
sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/$TASK/${split}.json"
DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/$TASK"

if [[ $TASK == "nq" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/${split}.json"
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ"
fi

if [[ $TASK == "tqa" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA/${split}.json"
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA"
fi

if [[ $TASK == att* ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx//data/att/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/data/att/$TASK/${split}.json"
fi

