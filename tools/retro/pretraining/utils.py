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

import os

from megatron import get_retro_args

# >>>
from lutil import pax
# <<<


# def get_base_nbr_workdir(args):
#     return os.path.join(args.retro_workdir, "nbr")
# def get_base_pretraining_workdir(args):
#     return os.path.join(args.retro_workdir, "pretraining")
def get_pretraining_workdir():
    args = get_retro_args()
    return os.path.join(args.retro_workdir, "pretraining")
