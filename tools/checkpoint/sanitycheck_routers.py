import torch
import glob
import os

path = '/home/yihuih/llmservice/fixrouter/gpt3-8x2b_TP8_init001/iter_1417624'

partitions = glob.glob(os.path.join(path, '*'))
a = torch.load(os.path.join(partitions[0], 'model_optim_rng.pt'))
print('using', partitions[0], 'as reference')
for partition in partitions[1:]:
    b = torch.load(os.path.join(partition, 'model_optim_rng.pt'))
    print('checking', partition)
    for k,v in a['model'].items():
        if 'router' in k:
            print('checking', k)
            assert torch.equal(a['model'][k], b['model'][k])