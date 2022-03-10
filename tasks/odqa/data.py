
import json
import os.path
from pathlib import Path



def load_data(data_path=None, with_context=False):
    assert data_path
    if data_path.endswith(('.jsonl', 'txt')):
        data = open(data_path, 'r')
    elif data_path.endswith(('.json')):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    
    examples = []
    for k, example in enumerate(data):
        if data_path is not None and data_path.endswith(('.jsonl', 'txt')):
            example = json.loads(example)
        new_example = {}
        new_example['id'] = k
        new_example['question'] = example['question']
        if 'answers' in example:
            new_example['answers'] = example['answers']
        elif 'answer' in example:
            new_example['answers'] = example['answer']
        if 'target' in example:
            new_example['target'] = example['target']
        if with_context:
            if 'ctxs' in example:
                new_example['ctxs'] = example['ctxs'][0]
            else:
                new_example['ctxs'] = 'no context'
        examples.append(new_example)

    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()
    return examples

def load_data_distributed(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith(('.json')):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    print("the global_rank is {}, and the world_size is {}".format(global_rank, world_size))
    for k, example in enumerate(data):
        if k > 16:
            break
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        examples.append(example)
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


def load_piQA_data(data_path=None):
    assert data_path
    lable_file_path = Path(os.path.dirname(data_path)) / os.path.basename(data_path).replace('.jsonl', '-labels.lst')
    raw_data = open(data_path, 'r')
    with open(lable_file_path, 'r') as f:
        raw_lable_data = f.readlines()

    examples = []
    new_data={}
    for each_data, each_lable in zip(raw_data, raw_lable_data):
        new_data = json.loads(each_data)
        new_data['golden'] = each_lable
        examples.append(new_data)
    
    return examples



