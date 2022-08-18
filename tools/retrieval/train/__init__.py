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

# def run_train_pipeline(args, timer):
def train_index(args, timer):

    assert torch.cuda.is_available(), "index requires cuda."

    # Init index.
    timer.push("init")
    index = IndexFactory.get_index(args)
    timer.pop()

    # Train index.
    timer.push("train")
    index.train(args.train_paths, args.index_dir_path, timer)
    timer.pop()
