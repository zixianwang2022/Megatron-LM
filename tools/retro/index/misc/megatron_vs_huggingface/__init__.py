# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .v0 import run_bert_comparison as run_bert_comparison_v0
from .v1 import run_bert_comparison as run_bert_comparison_v1
from .v2 import run_bert_comparison as run_bert_comparison_v2
from .v3_full_db import run_bert_comparison as run_bert_comparison_v3_full_db
from .v4_partial_db import run_bert_comparison as run_bert_comparison_v4_partial_db
from .v5_dist_comp import run_bert_comparison as run_bert_comparison_v5_dist_comp
