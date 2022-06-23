
# CGAP: Context Generation and Answer Prediction for ODQA

#### Step 1: Prepare the Dataset.
You can download NQ and TriviaQA datasets via [FiD] (https://github.com/facebookresearch/FiD)

Download the WebQuestion dataset from [DrQA]():
<pre>
    git clone https://github.com/facebookresearch/DrQA.git
    cd DrQA
    export DRQA_PATH=$(pwd)
    sh download.sh
</pre>
Put the data under your `<DATA_DIR>`

#### Step 2: Run CGAP

* If you are running the LM on local machine, use
` sh examples/odqa/cgap.sh`

* If you are calling the model via API (e.g. the 8.3B model), use
`sh examples/odqa/cgap_api.sh`

#### Step 3: Run Evaluation
`sh examples/odqa/evaluate.sh`





