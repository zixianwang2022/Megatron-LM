

## Prompting large LMs for open-domain QA

### Step 1. download the data (NQ and TriviaQA)

We use the scripts from Fid to download the NQ and TriviaQA, and use the same data split as them.

```sh get-data.sh```

### Step 2. Preproces the downloaded data ?


### Step 3. Few-shot Prompting LMs for the answer

<!-- input:  <Q_1, A_1, ... Q_k, A_k, Q>, k=64,  output: A -->

### Step 4. Evalulate