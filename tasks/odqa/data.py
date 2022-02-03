
import random
import json
import torch



def load_data(data_path=None):
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
        new_example['question'] = example['question']
        if 'answers' in example:
            new_example['answers'] = example['answers']
        elif 'answer' in example:
            new_example['answers'] = example['answer']
        if 'target' in example:
            new_example['target'] = example['target']
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 question_prefix='Question:',
                 answer_prefix='Answer:'):
        self.data = data
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

class Collator(object):
    def __init__(self, text_maxlength, answer_maxlength=20):
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)
