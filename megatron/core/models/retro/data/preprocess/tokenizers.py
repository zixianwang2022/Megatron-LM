# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass
from typing import Any


@dataclass
class RetroTokenizers:

    gpt: Any = None
    bert: Any = None
