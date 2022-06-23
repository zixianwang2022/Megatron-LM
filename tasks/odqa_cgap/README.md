# CGAP: Context Generation and Answer Prediction for Open-domain Question Answering

Below we present the steps to run our CGAP API.

## CGAP-API

### Data Preparation
To run the cgap api, you need to prepare a database so that we can select the prompt samples for the given question. 
You can either use NaturalQuestions or TriviaQA training data as the prompt file.
You can use the scripts from the [FiD github](https://github.com/facebookresearch/FiD) to download the corresponding training data. 

Create your data dirctory:
`mkdir DATASET_FOLDER`
Put the downloaded prompt file under corresponding directory as:
`${DATASET_FOLDER}/NQ/train.json` and `${DATASET_FOLDER}/TQA/train.json`
You also need to provide the file to save the prompt data embeddings, so that the retriever will load the embeddings directly next time for faster retrieval.

### Call the API via Python COMMAND
`sh api_cgap.sh`

### Start the CGAP API Demo Server
`sh start_server.sh`
