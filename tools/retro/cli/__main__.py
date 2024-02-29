# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os

from . import retro

if __name__ == "__main__":
    # >>>
    # retro.init(os.environ["RETRO_WORKDIR"])
    retro.init("/lustre/fs6/portfolios/adlr/users/lmcafee/retro/projects/wiki-core-bert-mlm")
    # <<<
