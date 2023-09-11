
FEWSHOT_FOLDER_SINGLE_TURN="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa"
FEWSHOT_FOLDER_MULTI_TURN="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa"


FEWSHOT_FILES="drop ${FEWSHOT_FOLDER_SINGLE_TURN}/drop/fewshot_samples.json \
NarrativeQA ${FEWSHOT_FOLDER_SINGLE_TURN}/NarrativeQA/fewshot_samples.json \
Quoref ${FEWSHOT_FOLDER_SINGLE_TURN}/Quoref/fewshot_samples.json \
ROPES ${FEWSHOT_FOLDER_SINGLE_TURN}/ROPES/fewshot_samples.json \
squad1.1 ${FEWSHOT_FOLDER_SINGLE_TURN}/squad1.1/fewshot_samples.json \
squad2.0 ${FEWSHOT_FOLDER_SINGLE_TURN}/squad2.0/fewshot_samples.json \
newsqa ${FEWSHOT_FOLDER_SINGLE_TURN}/newsqa/fewshot_samples.json \
convqa ${FEWSHOT_FOLDER_MULTI_TURN}/convqa/fewshot_samples.json \
chatgptgennoanswer ${FEWSHOT_FOLDER_MULTI_TURN}/chatgptgen/fewshot_samples.json \
quiet_cockatoo ${FEWSHOT_FOLDER_SINGLE_TURN}/quiet_cockatoo/fewshot_samples.json"
