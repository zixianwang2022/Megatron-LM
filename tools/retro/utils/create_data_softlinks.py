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

import glob
import os


def create_data_softlinks(global_dirs, local_root_dir):

    # Soft links. [ personal space ]
    for data_index, global_dir in enumerate(global_dirs):

        print("soft links, data %d / %d." % (data_index, len(global_dirs)))

        local_dir = os.path.join(
            local_root_dir,
            os.path.basename(global_dir)
        )

        global_files = [
            f
            for f in glob.glob(global_dir + "/*")
            if f.endswith(".bin") or f.endswith(".idx")
        ]

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        for global_file in global_files:
            local_file = os.path.join(local_dir, os.path.basename(global_file))
            if not os.path.exists(local_file):
                os.symlink(global_file, local_file)


if __name__ == "__main__":
    # . /mnt/fsx-outputs-chipdesign/plegresley/data/gpt3/gpt3_blend.sh
    # /mnt/fsx-outputs-chipdesign/plegresley/data/gpt3/

    global_prefixes = sorted([
        f
        for f in glob.glob("/mnt/fsx-outputs-chipdesign/plegresley/data/gpt3/*")
        if os.path.isdir(f)
    ])
    local_root_dir = "/mnt/fsx-outputs-chipdesign/lmcafee/retro/data"

    create_data_softlinks(global_prefixes, local_root_dir)
