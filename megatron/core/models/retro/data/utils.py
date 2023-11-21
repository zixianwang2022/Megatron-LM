# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig

from .external_libs import h5py


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def extract_data_config(config):
    return config.retro_gpt_datasets.train[0].config


def get_config_path(project_dir):
    '''Config copy stored within retro project dir.'''
    return os.path.join(project_dir, "config.json")


# >>>
# def get_num_chunks_per_sample(config):
#     '''Compute seq_length // chunk_length.'''
#     sample_length = config.retro_gpt_seq_length
#     chunk_length = config.retro_gpt_chunk_length
#     assert sample_length % chunk_length == 0
#     return sample_length // chunk_length
def get_num_chunks_per_sample(sample_length, chunk_length):
    '''Compute seq_length // chunk_length.'''
    assert sample_length % chunk_length == 0
    return sample_length // chunk_length
# <<<


# >>>
# def get_gpt_data_dir(config):
#     return os.path.join(config.retro_project_dir, "data")
def get_gpt_data_dir(project_dir):
    return os.path.join(project_dir, "data")
# <<<


def core_gpt_dataset_config_from_retro_preprocessing_config(
    config,
    is_dataset_built_on_rank,
    # >>>
    # return_document_ids,
    # <<<
):
    data_dir = get_gpt_data_dir(config.retro_project_dir)
    # >>>
    blend = list(config.retro_gpt_data_path)
    # blend = list(config.data_path)
    # <<<
    for i in range(len(blend) - 1, -1, -2):
        blend[i] = os.path.join(data_dir, blend[i])
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=config.retro_gpt_seed,
        sequence_length=config.retro_gpt_seq_length,
        blend=blend,
        split=config.retro_gpt_split,
        path_to_cache=config.retro_gpt_data_cache_path,
        # >>>
        # return_document_ids=return_document_ids,
        return_document_ids=True,
        # <<<
    )


class GPTToTextDataset(torch.utils.data.Dataset):
    '''Dataset to convert GPT tokens to text.'''

    def __init__(self, gpt_dataset, gpt_tokenizer):

        super().__init__()

        self.gpt_dataset = gpt_dataset
        self.gpt_tokenizer = gpt_tokenizer

    def __len__(self):
        return len(self.gpt_dataset)

    def __getitem__(self, idx):
        gpt_token_ids = self.gpt_dataset[idx]["text"].tolist()
        text = self.gpt_tokenizer.detokenize(gpt_token_ids)
        return {"text": text}


# >>>
# def save_data(data_map, *args):
#     '''Save map of numpy arrays to hdf5 file.'''
#     # >>>
#     raise Exception("hi.")
#     # <<<

#     # Parse args.
#     if len(args) == 1:
#         path = args[0]
#     elif len(args) == 2:
#         dir_path, file_name = args
#         path = os.path.join(dir_path, file_name)
#     else:
#         raise Exception("specialize for len(args) == %d." % len(args))

#     # Save data.
#     if not os.path.isfile(path):
#         f = h5py.File(path, "w")
#         for k, v in data_map.items():
#             f.create_dataset(k, data=v)
#         f.close()

#     return path


# def load_data(paths):
#     '''Load multiple hdf5 files to single numpy array.'''
#     # >>>
#     raise Exception("hi.")
#     # <<<

#     # Read data shapes.
#     shape_map = defaultdict(lambda : (0, None))
#     for p in paths:
#         f = h5py.File(p, "r")
#         for k in f.keys():
#             shape = tuple(f[k].shape)
#             shape_map[k] = (shape_map[k][0] + shape[0], shape[1])
#         f.close()

#     # Allocate output array.
#     data_map = { k : np.empty(s, dtype="f4") for k, s in shape_map.items() }
#     start_map = { k : 0 for k in shape_map }

#     # Load files.
#     for pi, p in enumerate(tqdm(paths, "load data")):
#         f = h5py.File(p, "r")
#         for k in f.keys():
#             i0 = start_map[k]
#             i1 = i0 + len(f[k])
#             data_map[k][i0:i1] = f[k]
#             start_map[k] += len(f[k])
#         f.close()

#     return data_map
# <<<


def get_missing_blocks(project_dir, n_samples, block_size,
                       validate=lambda f : None):
    '''Divide range [0, num_samples) to sequence of block ranges.

    This is a core method within the concept of block processing. The idea
    is to divide a range (size n_samples) into a sequence of blocks. Each
    block corresponds to a file within 'project_dir' with name
    '{start_idx}-{end_idx}.hdf5'. This method checks for the existence of
    these files, and returns a list of the ones that are missing.
    '''

    # Block ranges.
    block_start_idxs = list(range(0, n_samples, block_size))
    block_end_idxs = [ min(n_samples, i + block_size) for i in block_start_idxs ]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # All block files (existing + missing).
    n_digits = int(np.ceil(np.log(n_samples) / np.log(10)) + 1)
    all_blocks = [{
        "range" : r,
        "path" : os.path.join(
            project_dir,
            "%s-%s.hdf5" % tuple([ str(i).zfill(n_digits) for i in r ]),
        )
    } for r in block_ranges]
    all_block_path_set = set(block["path"] for block in all_blocks)

    # Delete corrupt files.
    if torch.distributed.get_rank() == 0:
        existing_block_paths = [block["path"]
                                for block in all_blocks
                                if os.path.exists(block["path"])]
        for index, path in enumerate(
                tqdm(existing_block_paths, "validating block.")):

            assert path in all_block_path_set, "unexpected filename, '%s'." % path

            try:
                f = h5py.File(path, "r")
            except:
                # raise Exception("unable to open/validate '%s'." % path)
                os.remove(path)
                continue

            try:
                validate(f)
            except:
                # raise Exception("delete block file '%s'." % path)
                os.remove(path)
            finally:
                f.close()

    # Wait for files to be deleted.
    torch.distributed.barrier()

    # Filter missing files.
    missing_blocks = [block
                      for block in all_blocks
                      if not os.path.exists(block["path"])]

    return missing_blocks


def get_missing_blocks_by_rank(project_dir, n_samples, block_size,
                               validate=lambda f : None):
    '''Divide missing blocks evenly across all ranks.

    See 'get_missing_blocks()' above for description. The returned list of
    missing blocks is split evenly across ranks via interleaving. This way,
    each rank has a roughly equal number of blocks to process for a
    downstream operation.
    '''

    missing_blocks = get_missing_blocks(project_dir, n_samples, block_size,
                                        validate)

    # This rank's missing files.
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    data_parallel_world_size = parallel_state.get_data_parallel_world_size()
    rank_missing_blocks = missing_blocks[data_parallel_rank:len(missing_blocks):data_parallel_world_size]

    # Extend rank's missing blocks (with None) such that all ranks have equal
    # length lists. This allows for easier tracking of global progress.
    n_missing_tensor = torch.cuda.LongTensor([len(rank_missing_blocks)])
    torch.distributed.all_reduce(n_missing_tensor,
                                 op=torch.distributed.ReduceOp.MAX)
    max_n_missing = n_missing_tensor.item()
    rank_missing_blocks += [None] * (max_n_missing - len(rank_missing_blocks))

    return len(missing_blocks), rank_missing_blocks


class BlockPathMap:
    '''Map an index to its containing block path.

    The common use for this class is to have a directory of files containing
    blocks of processed data, of uniform block size (e.g., 100k samples per
    file). Each file must follow a naming convention of 'startIdx-endIdx.[ext]',
    where 'endIdx' minus 'startIdx' must equal the block size, with the possible
    exception of the final block. Given an input index, this class maps the
    index to the containing block file.
    '''

    @classmethod
    def from_dir(cls, _dir, block_size, ext="hdf5"):
        '''Get list of block files, and create map.'''
        assert os.path.isdir(_dir), f"directory not found, '{_dir}'."
        return cls(sorted(glob.glob(_dir + f"/*.{ext}")), block_size)

    def __init__(self, block_paths, block_size):
        self.max_idx = 0
        self.block_path_map = {}
        for block_path in block_paths:
            name = os.path.splitext(os.path.basename(block_path))[0]
            start_idx, end_idx = [ int(i) for i in name.split("-") ]
            self.block_path_map[start_idx] = block_path
            self.max_idx = max(self.max_idx, end_idx)
        self.block_size = block_size

    def __str__(self):
        return "%d paths" % len(self.block_path_map)

    def __getitem__(self, idx):
        '''Get block path from index.'''
        block_start_idx = self.block_size * (idx // self.block_size)
        block_path = self.block_path_map[block_start_idx]
        return block_path
