# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from .v0 import run_bert_comparison as run_bert_comparison_v0
from .v1 import run_bert_comparison as run_bert_comparison_v1
from .v2 import run_bert_comparison as run_bert_comparison_v2
from .v3_full_db import run_bert_comparison as run_bert_comparison_v3_full_db
from .v4_partial_db import run_bert_comparison as run_bert_comparison_v4_partial_db
from .v5_dist_comp import run_bert_comparison as run_bert_comparison_v5_dist_comp
