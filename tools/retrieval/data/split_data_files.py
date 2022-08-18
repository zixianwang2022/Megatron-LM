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

def split_data_files():

    raise Exception("split again?")

    # ~~~~~~~~ feat paths [ hdf5 ] ~~~~~~~~
    input_dir = os.path.join(args.base_dir, "enwiki-feat-16")
    output_dir = os.path.join(args.base_dir, "enwiki-feat-16-split")
    input_paths = glob.glob(os.path.join(input_dir, "Wikipedia*.feat.hdf5"))

    batch_size = int(1e5)

    # ~~~~~~~~ load feats ~~~~~~~~
    for input_index, input_path in enumerate(input_paths):

        # >>>
        # if i == 1:
        #     break
        # <<<

        print("load input feats %d / %d." % (input_index, len(input_paths)))

        finput = h5py.File(input_path, "r")
        input_data = finput["feat"]
        ninput = len(input_data)
        for output_index, start_index in enumerate(range(0, ninput, batch_size)):

            print("  %d / %d." % (output_index, int(np.ceil(ninput/batch_size))))

            end_index = min(ninput, start_index + batch_size)
            output_data = input_data[start_index:end_index]

            output_path = os.path.join(
                output_dir,
                "%d-%d.hdf5" % (input_index, output_index),
            )
            foutput = h5py.File(output_path, "w")
            foutput.create_dataset("feat", data = output_data)
            foutput.close()

    raise Exception("split.")
