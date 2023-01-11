# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import gc
import time
import torch

from tools.retro.utils import Timer


def test_monolithic_alloc_performance():

    assert torch.distributed.get_rank() == 0

    def time_alloc(n):
        timer = Timer()

        timer.push(f"n {n}")
        data = np.empty((n, 1024), dtype = "f4")
        data.fill(0)
        timer.pop()

        del data
        gc.collect()

    for _pow in range(9):
        n = int(np.power(10, _pow))
        time_alloc(n)
    time_alloc(int(150e6))
    time_alloc(int(200e6))
    time_alloc(int(250e6))
    time_alloc(int(300e6))


def test_iterative_alloc_performance():

    assert torch.distributed.get_rank() == 0

    n_feats = 1024
    n_samples = 300000000
    # block_size = 1000000
    block_size = 3750000 # *
    # block_size = 10000000

    data = np.empty((n_samples, n_feats), dtype = "f4")
    # data.fill(0) # ... allocates 1.2TB for real; *essential* for performance

    start_time = time.time()
    for block_start_idx in range(0, n_samples, block_size):

        block_end_idx = min(n_samples, block_start_idx + block_size)
        block = np.zeros((block_end_idx - block_start_idx, n_feats), dtype = "f4")
        data[block_start_idx:block_end_idx] = block

        elapsed_time = time.time() - start_time
        print("block %d / %d ... %.1f min, %.1f min." % (
            block_start_idx // block_size,
            int(np.ceil(n_samples / block_size)),
            elapsed_time / 60,
            elapsed_time * n_samples / block_end_idx / 60,
        ))
