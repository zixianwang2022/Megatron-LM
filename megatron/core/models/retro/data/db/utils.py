# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import glob
import json
import numpy as np
import os

# >>>
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.models.retro.data.external_libs import h5py

from .dataset import DBDataset
# <<<

def get_base_db_workdir(env):
    '''Sub-directory for DB data.'''
    return os.path.join(env.config.retro_workdir, "db")


def get_indexed_dataset_infos_path(env):
    '''Path to indexed dataset meta-infos.'''
    return os.path.join(get_base_db_workdir(env), "indexed_dataset_infos.json")


def save_indexed_dataset_infos(env, indexed_dataset_infos):
    '''Save dataset order & meta-info.'''

    # Remove 'dataset' field.
    clean_infos = []
    for info in indexed_dataset_infos:
        info = dict(info)
        del info["dataset"]
        clean_infos.append(info)

    # Save.
    with open(get_indexed_dataset_infos_path(env), "w") as f:
        json.dump(clean_infos, f, indent=4)


def get_indexed_dataset_infos(env):
    '''Load indexed dataset meta-infos.'''

    # Load json.
    path = get_indexed_dataset_infos_path(env)
    with open(path) as f:
        infos = json.load(f)

    # Add indexed datasets.
    for info in infos:
        info["dataset"] = MMapIndexedDataset(info["prefix"])

    return infos


def get_individual_db_dir(env, name):
    '''Individual DB's directory.'''
    return os.path.join(get_base_db_workdir(env), "individual", name)


def get_individual_chunk_db(env, ds_id, ds_info):
    '''Load individual dataset's chunk DB.'''
    db_paths = sorted(glob.glob(ds_info["db_dir"] + "/*hdf5"))
    # *Note*: convert to dataset, rather than copying to memory.
    db = np.zeros((ds_info["n_chunks"], 5), dtype="uint32")
    db[:, 0] = ds_id
    start_idx = 0
    for db_path in db_paths:
        f = h5py.File(db_path, "r")
        n_chunks_current = f["chunks_valid"].shape[0]
        db[start_idx:(start_idx+n_chunks_current), 1:] = f["chunks_valid"]
        start_idx += n_chunks_current
        f.close()

    assert start_idx == ds_info["n_chunks"]

    return db


def get_individual_doc_offsets(env, ds_id, ds_info):
    '''Load individual dataset's chunk DB.'''
    paths = sorted(glob.glob(ds_info["db_dir"] + "/*hdf5"))
    # *Note*: convert to dataset, rather than copying to memory.
    doc_offsets = np.zeros((ds_info["n_docs"], 3), dtype="uint64")
    doc_offsets[:, 0] = ds_id
    start_idx = 0
    start_offset = 0
    for path in paths:
        with h5py.File(path) as f:
            current_doc_offsets = np.copy(f["doc_offsets"])
            current_doc_offsets[:, 1] += start_offset
            current_ndocs = current_doc_offsets.shape[0]
            doc_offsets[start_idx:(start_idx+current_ndocs), 1:] = \
                current_doc_offsets
            start_idx += current_ndocs
            start_offset = current_doc_offsets[-1, 1].item()

    return doc_offsets


def get_merged_db_path_map(env):
    '''Paths to merged datasets.'''
    base_dir = get_base_db_workdir(env)
    return {
        "sampled" : os.path.join(base_dir, "merged", "sampled.hdf5"),
        "train" : os.path.join(base_dir, "merged", "train.hdf5"),
        "valid" : os.path.join(base_dir, "merged", "valid.hdf5"),
    }


def get_merged_dataset(env, db_type, indexed_dataset_infos=None):
    '''Get merged dataset.'''

    if not indexed_dataset_infos:
        indexed_dataset_infos = get_indexed_dataset_infos(env)

    # Load chunks.
    db_path = get_merged_db_path_map()[db_type]
    f = h5py.File(db_path, "r")
    chunks = f["chunks"]

    # DB dataset.
    indexed_datasets = [ info["dataset"] for info in indexed_dataset_infos ]
    dataset = DBDataset(db_path, indexed_datasets, chunks,
                        args.retro_gpt_chunk_length)

    return dataset


def get_merged_sampled_dataset(env, indexed_dataset_infos=None):
    return get_merged_dataset(env, "sampled", indexed_dataset_infos)


def get_merged_train_dataset(env, indexed_dataset_infos=None):
    return get_merged_dataset(env, "train", indexed_dataset_infos)


def get_merged_valid_dataset(env, indexed_dataset_infos=None):
    return get_merged_dataset(env, "valid", indexed_dataset_infos)
