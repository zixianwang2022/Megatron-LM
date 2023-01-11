# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

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

    global_prefixes = sorted([
        f
        for f in glob.glob("/from/dir/*")
        if os.path.isdir(f)
    ])
    local_root_dir = "/to/dir"

    create_data_softlinks(global_prefixes, local_root_dir)
