import torch
import os
import random

import json
from time import sleep

from triton.runtime.cache import FileCacheManager


def get_rank():
    return torch.distributed.get_rank()


class ParallelFileCacheManager(FileCacheManager):

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = random.randint(0, 1000000)
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()

        # Note (rwaleffe): this barrier prevents one rank from falling behind and noticing the filepath before entering
        # this function (causing a hang on the second barrier due to not entering this function at all)
        # Initial dist barrier
        # torch.distributed.barrier()

        # Write only on rank 0. This prevents FileNotFound errors while compiling triton kernels in parallel
        # (e.g., distributed training).
        # if get_rank() == 0:

        # use tempfile to be robust against program interruptions
        temp_path = f"{filepath}.tmp.pid_{pid}_{rnd_id}"
        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        # Note (rwaleffe): try except needed on distributed file systems
        try:
            os.replace(temp_path, filepath)
        except:
            pass
            # raise FileNotFoundError(f"FILE NOT FOUND")

        # Final dist barrier
        # torch.distributed.barrier()

        return filepath