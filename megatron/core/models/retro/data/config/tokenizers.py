# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

'''Container class for GPT and Bert tokenizers.'''

from dataclasses import dataclass
from typing import Any


@dataclass
class RetroTokenizers:
    '''Container class for GPT and Bert tokenizers.'''

    gpt: Any = None
    bert: Any = None
