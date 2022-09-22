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

def create_data_softlinks(data_files):

    # Soft links. [ personal space ]
    root_dir = \
        "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/preprocess/data"
    for data_index, global_file in enumerate(data_files):

        print("soft links, data %d / %d." % (data_index, len(data_files)))

        local_dir = os.path.join(
            root_dir,
            os.path.basename(os.path.dirname(global_file)),
        )
        local_prefix = os.path.join(
            local_dir,
            os.path.splitext(os.path.basename(global_file))[0],
        )
        global_prefix = os.path.splitext(global_file)[0]

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        for ext in [ "bin", "idx" ]:
            local_file = local_prefix + "." + ext
            if not os.path.exists(local_file):
                os.symlink(global_prefix + "." + ext, local_file)

        # pax(0, {
        #     "global_file" : global_file,
        #     "root_dir" : root_dir,
        #     "local_dir" : local_dir,
        #     "local_prefix" : local_prefix,
        #     "global_prefix" : global_prefix,
        # })

    pax(0, {"data_files": data_files})
    # raise Exception("soft link.")
