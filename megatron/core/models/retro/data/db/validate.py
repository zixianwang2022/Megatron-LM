# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

? ? ?

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import torch
from typing import List

from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.models.retro.data.config import RetroPreprocessingConfig
from megatron.core.models.retro.data.external_libs import h5py
from megatron.core.models.retro.data.utils import (
    get_sampled_blocks_by_rank,
    print_rank_0,
)

from .build import build_block_db
from .utils import get_indexed_dataset_infos, get_merged_datasets

# >>>
from lutil import pax, print_seq
# <<<


def validate_individual_db(
    config: RetroPreprocessingConfig,
    dataset_idx: int,
    n_datasets: int,
    dataset_info: dict,
) -> None:
    '''Process a single indexed dataset & extract chunks.'''

    # Make directory.
    db_dir = dataset_info["db_dir"]

    # Indexed dataset.
    indexed_dataset = dataset_info["dataset"]

    # Sample existing DB blocks (split by documents).
    blocks = get_sampled_blocks_by_rank(
        db_dir,
        len(indexed_dataset),
        config.retro_doc_block_size,
        validate=lambda f : f["chunks_valid"].shape == (0,) \
            or f["chunks_valid"].shape[1] == 4,
        fraction=config.retro_task_validate,
    )

    assert blocks.n_missing_world == 0

    # >>>
    # print_seq("sampled blocks = %d ... %s." % (
    #     len(blocks.existing),
    #     [ b["range"] for b in blocks.existing ],
    # ))
    # <<<

    # Num processes.
    n_procs = 8

    # Process documents in parallel.
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for block_idx, block in enumerate(blocks.existing):

            if block is not None:

                # Load existing block DB.
                with h5py.File(block["path"]) as f:
                    existing_chunks_valid = np.copy(f["chunks_valid"])
                    existing_chunks_invalid = np.copy(f["chunks_invalid"])
                    existing_doc_offsets = np.copy(f["doc_offsets"])

                # Build new block DB.
                chunks_valid, chunks_invalid, doc_offsets = build_block_db(
                    config=config,
                    dataset_idx=dataset_idx,
                    n_datasets=n_datasets,
                    indexed_dataset=indexed_dataset,
                    n_procs=n_procs,
                    executor=executor,
                    n_missing_blocks=len(blocks.existing),
                    block_idx=block_idx,
                    block=block,
                )

                # Check equality.
                assert np.array_equal(existing_chunks_valid, chunks_valid)
                assert np.array_equal(existing_chunks_invalid, chunks_invalid)
                assert np.array_equal(existing_doc_offsets, doc_offsets)

                # >>>
                # pax(
                #     "block",
                #     "existing_chunks_valid, existing_chunks_invalid, existing_doc_offsets",
                #     "chunks_valid, chunks_invalid, doc_offsets",
                # )
                # <<<

            # Wait for all ranks to finish block.
            print_rank_0(" > waiting for all ranks to finish block.")
            torch.distributed.barrier()

    print_rank_0(" > finished validating individual db.")


def validate_individual_dbs(
    config: RetroPreprocessingConfig,
    indexed_dataset_infos: List[dict],
) -> None:
    '''Iterate each indexed dataset & process its chunks.'''

    # Validate individual DBs.
    print_rank_0(" > validate individual chunk dbs.")
    for ds_idx, ds_info in enumerate(indexed_dataset_infos):

        # Progress.
        print_rank_0(" > validating individual db, dataset %d / %d ... '%s'." % (
            ds_idx,
            len(indexed_dataset_infos),
            ds_info["name"],
        ))

        # Process single dataset.
        validate_individual_db(config, ds_idx, len(indexed_dataset_infos), ds_info)


# >>>
# def validate_db(config):

#     # train_dataset = get_merged_train_dataset(
#     merged_ds_map = get_merged_datasets(
#         project_dir=config.retro_project_dir,
#         chunk_length=config.retro_gpt_chunk_length,
#         eod_token_id=config.retro_tokenizers.gpt.eod,
#     )

#     pax("config, merged_ds_map")


# def validate_merged_dbs(project_dir, indexed_dataset_infos):
#     validate_merged_db(project_dir, indexed_dataset_infos, "sampled")
#     validate_merged_db(project_dir, indexed_dataset_infos, "train")
#     validate_merged_db(project_dir, indexed_dataset_infos, "valid")
# <<<


def validate_db(config):
    '''Extract token chunks from each indexed dataset.

    Iterate each document of each indexed dataset, extract that document's
    chunks, and save to a 'DB' (hdf5 file).
    '''

    project_dir = config.retro_project_dir

    # Indexed dataset info.
    indexed_dataset_infos = get_indexed_dataset_infos(project_dir)

    # Validate individual dbs.
    validate_individual_dbs(config, indexed_dataset_infos)

    ########################################################
    # [ **Note**: individual checks considered sufficient. ]
    ########################################################

    # # Single-process going forward.
    # if torch.distributed.get_rank() != 0:
    #     return

    # # Validate merged dbs.
    # validate_merged_dbs(project_dir, indexed_dataset_infos)
