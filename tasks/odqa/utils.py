
import os
from pathlib import Path
import json
from tqdm import tqdm
import os.path
import string
import unicodedata
from typing import Tuple, List, Dict



def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    ans_dict = {}
    for path in files:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                id, answer = line.split('\t')
                ans_dict[int(id)] = answer
        path.unlink()

    with open(output_path, 'w') as outfile:
        for id, ans in sorted(ans_dict.items()):
            outfile.write(ans)
            
    glob_path.rmdir()
    
    return True


def result_analysis(prediction_file, ground_truth_file, gen_ctx_file):

    # assert ground_truth_file.endswith('json'), "the ground truth file should be the original json file" 

    gen_ctx_list = []

    with open(gen_ctx_file, 'r') as f:
        gen_ctx_list = f.readlines()

    prediction_list = []
    print('reading %s' % prediction_file)
    with open(prediction_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.replace("Answer:","")
            line = line.replace("Answer: ","")
            line = line.replace('????  ', "")
            line = line.replace('A: ',"")
            line = line.replace("A:", "")

            line = line.strip()

            if "<|endoftext|>" in line:
                line = line.replace("<|endoftext|>", "")
            prediction_list.append(line)

    ground_truths_list = []
    
    if ground_truth_file.endswith(('txt', 'lst')):
        raw_data = open(ground_truth_file, 'r')
    else:
        with open(ground_truth_file, 'r') as f:
            raw_data = json.load(f)

    for each in raw_data:
        if ground_truth_file.endswith('txt'):
            each = json.loads(each)
        
        if 'answers' in each:
            ground_truths_list.append(each['answers'])
        elif 'answer' in each:
            ground_truths_list.append(each['answer'])
        else:
            ground_truths_list.append([each])
    
    true_list = []
    false_list = []

    exactmatch = []
    for i,each in enumerate(prediction_list):
        print("=============")
        print(each)
        print(ground_truths_list[i])
        score = ems(each, ground_truths_list[i])
        print(score)
        exactmatch.append(score)
        if score:
            true_list.append(i)
        else:
            false_list.append(i)
    
    #
    analysis_true_data = []
    analysis_false_data = []
    all_data = []
    analysis_data = {}
    for i in range(len(prediction_list)):
        analysis_data = {}
        analysis_data['question'] = raw_data[i]['question']
        analysis_data['golden_ctx'] = raw_data[i]['ctxs'][0]
        if 'answer' in raw_data[i]: 
            analysis_data['golden_ans'] = raw_data[i]['answer'] 
        elif 'answers' in raw_data[i]:
           analysis_data['golden_ans'] = raw_data[i]['answers']
        else:  
            analysis_data['golden_ans'] = raw_data[i]['target'] 
        analysis_data['gen_ctx'] = gen_ctx_list[i]
        analysis_data['gen_ans'] = prediction_list[i]

        all_data.append(analysis_data)

        if i < 10:
            print(analysis_data)

        if i in true_list:
            analysis_true_data.append(analysis_data)
        else:
            analysis_false_data.append(analysis_data) 
    
    save_result_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.txt.all')
    with open(save_result_path, 'w') as f:
        json.dump(all_data, f, indent=4)

    save_true_result_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.txt.true')
    save_false_result_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.txt.false')


    with open(save_true_result_path, 'w') as f:
        json.dump(analysis_true_data, f, indent=4)
    print("save the true file done!")
    with open(save_false_result_path, 'w') as f:
        json.dump(analysis_false_data, f, indent=4)
    print("save the false file done!")

    save_true_list_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.true_list.txt')
    save_false_list_path = Path(os.path.dirname(gen_ctx_file)) / os.path.basename(gen_ctx_file).replace('.txt', '.false_list.txt')

    with open(save_true_list_path, 'w') as f:
        for each in true_list:
            f.write("%s\n" % each)

    with open(save_false_list_path, 'w') as f:
        for each in false_list:
            f.write("%s\n" % each)

    print("save the true list and false_list to file done!")

    final_em_score = np.mean(exactmatch)
   
    print('Exact Match: %.4f;' % final_em_score)

    print('done :-)')

    return final_em_score


def check_answer(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (doc_ids, doc_scores) = questions_answers_docs

    global dpr_all_documents
    hits = []

    for i, doc_id in enumerate(doc_ids):
        doc = dpr_all_documents[doc_id]
        text = doc[0]

        answer_found = False
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
        if match_type == "kilt":
            if has_answer_kilt(answers, text):
                answer_found = True
        elif has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits

def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


# function for the reader model answer validation
def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text):
    return unicodedata.normalize("NFD", text)


