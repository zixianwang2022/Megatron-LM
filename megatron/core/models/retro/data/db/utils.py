# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import glob
import json
import numpy as np
import os
from typing import List

from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.models.retro.data.config import RetroPreprocessingConfig
from megatron.core.models.retro.data.external_libs import h5py
from megatron.core.models.retro.data.utils import get_gpt_data_dir

from .dataset import DBDataset


def get_db_dir(project_dir):
    '''Sub-directory for DB data.'''
    return os.path.join(project_dir, "db")


def init_indexed_dataset_infos(config: RetroPreprocessingConfig) -> List[dict]:
    '''Gather meta-info about each indexed dataset.

    The returned info array allows for easy access to the configuration, and
    helps remove ambiguity.
    '''

    data_dir = get_gpt_data_dir(config.retro_project_dir)
    data_blend = config.retro_gpt_data_path
    assert len(data_blend) % 2 == 0, "currently, only blended dataset is supported."

    # Dataset infos.
    infos = []
    for i in range(0, len(data_blend), 2):
        ratio = float(data_blend[i])
        prefix = data_blend[i + 1]
        assert os.path.exists(os.path.join(data_dir, prefix + ".bin")), \
            "couldn't find '%s'." % path
        infos.append({
            "ratio" : ratio,
            "prefix" : prefix,
        })

    # Load indexed datasets.
    load_indexed_datasets(config.retro_project_dir, infos)

    return infos


def get_indexed_dataset_infos_path(project_dir):
    '''Path to indexed dataset meta-infos.'''
    return os.path.join(get_db_dir(project_dir), "indexed_dataset_infos.json")


def save_indexed_dataset_infos(project_dir, indexed_dataset_infos):
    '''Save dataset order & meta-info.'''

    # Remove 'dataset' field.
    clean_infos = []
    for info in indexed_dataset_infos:
        info = dict(info)
        del info["dataset"]
        clean_infos.append(info)

    # Save.
    with open(get_indexed_dataset_infos_path(project_dir), "w") as f:
        json.dump(clean_infos, f, indent=4)


def load_indexed_datasets(project_dir, indexed_dataset_infos):
    data_dir = get_gpt_data_dir(project_dir)
    for info in indexed_dataset_infos:
        info["dataset"] = MMapIndexedDataset(os.path.join(data_dir, info["prefix"]))


def get_indexed_dataset_infos(project_dir):
    '''Load indexed dataset meta-infos.'''

    # Load json.
    path = get_indexed_dataset_infos_path(project_dir)
    with open(path) as f:
        infos = json.load(f)

    # Load indexed datasets.
    load_indexed_datasets(project_dir, infos)

    return infos


def get_individual_db_dir(project_dir, prefix):
    '''Individual DB's directory.'''
    return os.path.join(get_db_dir(project_dir), "individual", prefix)


def get_individual_db_paths(project_dir, prefix):
    return sorted(glob.glob(get_individual_db_dir(project_dir, prefix) + "/*hdf5"))


def get_individual_chunk_db(project_dir, ds_id, ds_info):
    '''Load individual dataset's chunk DB.'''
    paths = get_individual_db_paths(project_dir, ds_info["prefix"])
    # *Note*: convert to dataset, rather than copying to memory.
    db = np.zeros((ds_info["n_chunks"], 5), dtype="uint32")
    db[:, 0] = ds_id
    start_idx = 0
    for path in paths:
        f = h5py.File(path, "r")
        n_chunks_current = f["chunks_valid"].shape[0]
        db[start_idx:(start_idx+n_chunks_current), 1:] = f["chunks_valid"]
        start_idx += n_chunks_current
        f.close()

    assert start_idx == ds_info["n_chunks"]

    return db


def get_individual_doc_offsets(project_dir, ds_id, ds_info):
    '''Load individual dataset's chunk DB.'''
    paths = get_individual_db_paths(project_dir, ds_info["prefix"])
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


def get_merged_db_path_map(project_dir):
    '''Paths to merged datasets.'''
    base_dir = get_db_dir(project_dir)
    return {
        "sampled" : os.path.join(base_dir, "merged", "sampled.hdf5"),
        "train" : os.path.join(base_dir, "merged", "train.hdf5"),
        "valid" : os.path.join(base_dir, "merged", "valid.hdf5"),
    }


def get_merged_dataset(project_dir, chunk_length, eod_token_id,
                       db_type, indexed_dataset_infos=None):
    '''Get merged dataset.'''

    if not indexed_dataset_infos:
        indexed_dataset_infos = get_indexed_dataset_infos(project_dir)

    # Load chunks.
    db_path = get_merged_db_path_map(project_dir)[db_type]
    f = h5py.File(db_path, "r")
    chunks = f["chunks"]

    # DB dataset.
    indexed_datasets = [ info["dataset"] for info in indexed_dataset_infos ]
    dataset = DBDataset(
        db_path=db_path,
        indexed_datasets=indexed_datasets,
        chunks=chunks,
        chunk_length=chunk_length,
        eod_token_id=eod_token_id,
    )

    return dataset


def get_merged_sampled_dataset(project_dir, chunk_length, eod_token_id,
                               indexed_dataset_infos=None):
    return get_merged_dataset(project_dir, chunk_length, eod_token_id,
                              "sampled", indexed_dataset_infos)


def get_merged_train_dataset(project_dir, chunk_length, eod_token_id,
                             indexed_dataset_infos=None):
    return get_merged_dataset(project_dir, chunk_length, eod_token_id,
                              "train", indexed_dataset_infos)


def get_merged_valid_dataset(project_dir, chunk_length, eod_token_id,
                             indexed_dataset_infos=None):
    return get_merged_dataset(project_dir, chunk_length, eod_token_id,
                              "valid", indexed_dataset_infos)


def get_merged_datasets(project_dir, chunk_length, eod_token_id):
    '''Get all merged datasets.'''
    fns = {
        "sampled" : get_merged_sampled_dataset,
        "train" : get_merged_train_dataset,
        "valid" : get_merged_valid_dataset,
    }
    datasets = { key : fn(project_dir, chunk_length, eod_token_id)
                 for key, fn in fns.items() }
    return datasets
