
import random
import json
import numpy as np



def load_data(data_path=None):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return data
