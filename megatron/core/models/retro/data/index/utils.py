# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import glob
import os


def get_index_dir(config):
    """Create sub-directory for this index."""

    # Directory path.
    index_dir_path = os.path.join(
        config.retro_project_dir,
        "index",
        config.retro_index_type,
        config.retro_index_str,
    )

    # Make directory.
    os.makedirs(index_dir_path, exist_ok=True)

    return index_dir_path


def num_samples_to_block_ranges(config, num_samples):
    '''Split a range (length num_samples) into sequence of block ranges
    of size block_size.'''
    block_size = config.retro_block_size
    start_idxs = list(range(0, num_samples, block_size))
    end_idxs = [min(num_samples, s + block_size) for s in start_idxs]
    ranges = list(zip(start_idxs, end_idxs))
    return ranges


def get_training_data_root_dir(config):
    return os.path.join(config.retro_project_dir, "index", "train_emb")


def get_training_data_block_dir(config):
    return os.path.join(get_training_data_root_dir(config), "blocks")


def get_training_data_block_paths(config):
    return sorted(glob.glob(get_training_data_block_dir(config) + "/*.hdf5"))


def get_training_data_merged_path(config):
    return os.path.join(
        get_training_data_root_dir(config),
        "train_%.3f.bin" % config.retro_index_train_load_fraction,
    )


def get_added_codes_dir(config):
    return os.path.join(get_index_dir(config), "add_codes")


def get_added_code_paths(config):
    return sorted(glob.glob(get_added_codes_dir(config) + "/*.hdf5"))
