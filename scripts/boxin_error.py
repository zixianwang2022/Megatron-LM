# lawrence mcafee

import time
import sys
# sys.path.append("/home/boxinw/megatron-lm-nvllm//megatron")
# sys.path.append("/home/boxinw/megatron-lm-nvllm/")

import h5py
import numpy as np
from tqdm import tqdm

import os, psutil

import argparse
import torch

# >>>
del os.environ["NCCL_DEBUG"]
# <<<

process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024 / 1024 / 1024)  # in bytes

from tools.retro.cli.db_cli import retro
retro.init("/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/next-llm")

for i in range(0, 19327461337, len(retro.db_dataset) // 10):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("gpt tokens  : %s" % str(retro.get_db_chunk_gpt(i)))
    print("bert tokens : %s" % str(retro.get_db_chunk_bert(i)))
    print("text        : %s" % str(retro.get_db_chunk_text(i)))

print(len(retro.db_dataset))


args = retro.args

from megatron import get_args, get_retro_args, mpu, print_rank_0

from tools.bert_embedding import BertEmbedder

# >>>
# tensor = torch.empty(
#     10,
#     10,
#     dtype=torch.uint8, # torch.int64,
#     # dtype="uint64",
#     device="cuda",
# )
# <<<

# >>>
if 0:
    print("~~~")
    print("params_dtype = %s." % args.params_dtype)
    exit()
# <<<

embedder = BertEmbedder(args.retro_bert_batch_size,
                        args.retro_bert_max_chunk_length,
                        args.bert_embedder_type)

# eof.
