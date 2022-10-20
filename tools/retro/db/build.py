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

from .embed import embed_db
from .preprocess import preprocess_db


# def build_db(args, timer):
# def build_chunk_db(args, timer):
def build_db(args, timer):

    # if torch.distributed.get_rank() != 0:
    #     return

    # # Preprocessing workdir.
    # workdir = os.path.join(args.retro_workdir, "preprocess")
    # os.makedirs(workdir, exist_ok = True)
    # workdir = args.retro_workdir

    # Stages.
    preprocess_chunk_db(args, timer)
    embed_chunk_db(args, timer)
