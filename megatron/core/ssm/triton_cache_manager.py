import torch
import os
import random
import json

from time import sleep
from pathlib import Path
from typing import Dict, Optional

from triton.runtime.cache import FileCacheManager


def get_rank():
    return torch.distributed.get_rank()


def default_cache_dir():
    return os.path.join(Path.home(), ".triton", "cache")


class ParallelFileCacheManager(FileCacheManager):
    def __init__(self, key):
        self.key = key
        self.lock_path = None
        # create cache directory if it doesn't exist
        self.cache_dir = os.environ.get('TRITON_CACHE_DIR', default_cache_dir())
        self.cache_dir = os.path.join(self.cache_dir, "rank_{}".format(get_rank()))
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)