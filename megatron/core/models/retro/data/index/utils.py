# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import glob
import os


def get_index_dir(env):
    """Create sub-directory for this index."""

    # Directory path.
    index_dir_path = os.path.join(
        env.config.retro_project_dir,
        "index",
        env.config.retro_index_type,
        env.config.retro_index_str,
    )

    # Make directory.
    os.makedirs(index_dir_path, exist_ok=True)

    return index_dir_path


def num_samples_to_block_ranges(env, num_samples):
    '''Split a range (length num_samples) into sequence of block ranges
    of size block_size.'''
    block_size = env.config.retro_block_size
    start_idxs = list(range(0, num_samples, block_size))
    end_idxs = [min(num_samples, s + block_size) for s in start_idxs]
    ranges = list(zip(start_idxs, end_idxs))
    return ranges


def get_training_data_root_dir(env):
    return os.path.join(env.config.retro_project_dir, "index", "train_emb")


def get_training_data_block_dir(env):
    return os.path.join(get_training_data_root_dir(env), "blocks")


def get_training_data_block_paths(env):
    return sorted(glob.glob(get_training_data_block_dir(env) + "/*.hdf5"))


def get_training_data_merged_path(env):
    return os.path.join(
        get_training_data_root_dir(env),
        "train_%.3f.bin" % env.config.retro_index_train_load_fraction,
    )


def get_added_codes_dir(env):
    return os.path.join(get_index_dir(env), "add_codes")


def get_added_code_paths(env):
    return sorted(glob.glob(get_added_codes_dir(env) + "/*.hdf5"))
