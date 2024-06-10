# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
import os
import random
import json
import socket

from time import sleep
from pathlib import Path
from typing import Dict, Optional

from triton.runtime.cache import FileCacheManager


def get_rank():
    return torch.distributed.get_rank()


def default_cache_dir():
    return os.path.join(Path.home(), ".triton", "cache")


class ParallelFileCacheManager(FileCacheManager):

    # See https://github.com/triton-lang/triton/blob/main/python/triton/runtime/cache.py

    # When running Triton with multiple ranks they each create their own cache manager. Their input
    # keys to that class are mostly (but not entirely) the same across ranks, which leads many ranks
    # to write to the same 'key' directories in the cache dir at the same time during compilation,
    # leading to conflicts.  This works around that by making each cache dir be rank specific by
    # adding "rank_<host>_<pid>" to the cache directory.

    def __init__(self, key):
        self.key = key
        self.lock_path = None
        # create cache directory if it doesn't exist
        self.cache_dir = os.environ.get('TRITON_CACHE_DIR', default_cache_dir())
        self.cache_dir = os.path.join(self.cache_dir, "rank_{}_{}".format(socket.gethostname(), os.getpid()))
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
