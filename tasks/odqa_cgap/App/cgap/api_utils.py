import json
import os.path
from pathlib import Path
import regex
import string

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

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

def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--nq-prompt-file', type=str, default=None,
                       help='prompting file')
    group.add_argument('--nq-encoded-ctx-file', type=str, default=None,
                       help='the encoded context file path storing the ctx embeddings')
    group.add_argument('--tqa-prompt-file', type=str, default=None,
                       help='prompting file')
    group.add_argument('--tqa-encoded-ctx-file', type=str, default=None,
                       help='the encoded context file path storing the ctx embeddings')

    group.add_argument('--num-prompt-examples', type=int, default=10,
                       help='number of prompt examples')
    group.add_argument('--out-seq-length', type=int, default=100,
                       help='output sequence length')
    group.add_argument('--megatron-api-url', type=str, default='http://10.14.74.235:5000/api',
                       help='url of the megatron api')
    group.add_argument('--top-p-sampling', type=float, default=0.0,
                       help='the top-p value')
    group.add_argument('--top-k-sampling', type=int, default=1,
                       help='the top-k value')
    group.add_argument('--temperature', type=float, default=1.0,
                       help='the temperature value')
    group.add_argument('--micro-batch-size', type=int, default=2,
                       help='the batch_size')
    group.add_argument('--random-seed', default=1234, type=int,
                       help='the random seed that megatron model used to generate text')
    group.add_argument('--db-name', default='NQ', type=str,
                       help='you can choose either NQ or TQA, as the backend prompting database')
    group.add_argument('--margin-number', default=4, type=int,
                       help='the number of marginalization contexts')
    group.add_argument('--ctx-length', default=64, type=int,
                       help='the length of tokens for your generated contexts')
    group.add_argument('--length-check', default=False, type=bool,
                       help='if set to True, will initiate a tokenizer, and truncate the input sequence if length > 2048')



    return parser.parse_args()




