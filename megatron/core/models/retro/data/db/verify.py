# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import torch
from typing import List

from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.models.retro.data.config import RetroPreprocessingConfig
from megatron.core.models.retro.data.utils import get_blocks_by_rank, print_rank_0

from .utils import (
    get_indexed_dataset_infos,
    # get_individual_db_dir,
    # get_individual_chunk_db,
    # get_individual_doc_offsets,
    get_merged_datasets,
    # get_merged_db_path_map,
    # load_indexed_datasets,
    # save_indexed_dataset_infos,
)

# >>>
from lutil import pax, print_seq
# <<<


def verify_individual_db(
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

    # Existing DB blocks (split by documents).
    blocks = get_blocks_by_rank(
        db_dir,
        len(indexed_dataset),
        config.retro_doc_block_size,
        validate=lambda f : f["chunks_valid"].shape == (0,) \
            or f["chunks_valid"].shape[1] == 4)

    assert blocks.n_missing_world == 0

    # Randomly sample blocks.
    n_blocks_sample = int(np.ceil(config.retro_task_verify * len(blocks.existing)))
    shuffled_blocks = [ b for b in blocks.existing if b is not None ]

    np.random.seed(None)
    np.random.shuffle(shuffled_blocks)

    shuffled_blocks = shuffled_blocks[:n_blocks_sample]
    shuffled_blocks += [None] * (n_blocks_sample - len(shuffled_blocks))

    # >>>
    print_seq("n_blocks_sample = %d / %d [%d] ... %s." % (
        n_blocks_sample,
        len(blocks.existing),
        len(shuffled_blocks),
        [ b["range"] for b in shuffled_blocks ],
    ))
    # <<<

    # Num processes.
    n_procs = 8
    # n_procs = 1 # ... for debug.

    # Process documents in parallel.
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for block_idx, block in enumerate(missing_db_blocks):

            if block is not None:

                # >>>
                pax("block")
                # <<<

                # Build block DB.
                chunk_db_valid, chunk_db_invalid, doc_offsets = build_block_db(
                    config=config,
                    dataset_idx=dataset_idx,
                    n_datasets=n_datasets,
                    # dataset_info=dataset_info,
                    indexed_dataset=indexed_dataset,
                    n_procs=n_procs,
                    executor=executor,
                    n_missing_blocks=len(missing_db_blocks),
                    block_idx=block_idx,
                    block=block,
                )

                # >>>
                pax("chunk_db_valid, chunk_db_invalid, doc_offsets")
                # <<<

            # Wait for all ranks to finish block.
            print_rank_0(" > waiting for all ranks to finish block.")
            torch.distributed.barrier()

    print_rank_0(" > finished saving individual db.")


def verify_individual_dbs(
    config: RetroPreprocessingConfig,
    indexed_dataset_infos: List[dict],
) -> None:
    '''Iterate each indexed dataset & process its chunks.'''

    # Verify individual DBs.
    print_rank_0(" > verify individual chunk dbs.")
    for ds_idx, ds_info in enumerate(indexed_dataset_infos):

        # Progress.
        print_rank_0(" > verifying individual db, dataset %d / %d ... '%s'." % (
            ds_idx,
            len(indexed_dataset_infos),
            ds_info["name"],
        ))

        # Process single dataset.
        verify_individual_db(config, ds_idx, len(indexed_dataset_infos), ds_info)


# >>>
# def verify_db(config):

#     # train_dataset = get_merged_train_dataset(
#     merged_ds_map = get_merged_datasets(
#         project_dir=config.retro_project_dir,
#         chunk_length=config.retro_gpt_chunk_length,
#         eod_token_id=config.retro_tokenizers.gpt.eod,
#     )

#     pax("config, merged_ds_map")
# <<<


def verify_db(config):
    '''Extract token chunks from each indexed dataset.

    Iterate each document of each indexed dataset, extract that document's
    chunks, and save to a 'DB' (hdf5 file).
    '''

    project_dir = config.retro_project_dir

    # Indexed dataset info.
    indexed_dataset_infos = get_indexed_dataset_infos(project_dir)

    # Verify individual dbs.
    verify_individual_dbs(config, indexed_dataset_infos)

    # Single-process going forward.
    if torch.distributed.get_rank() != 0:
        return

    # >>>
    # # Update n_chunks & save indexed dataset infos.
    # if not os.path.exists(get_indexed_dataset_infos_path(project_dir)):
    #     update_chunk_counts(config, indexed_dataset_infos)
    #     # >>>
    #     # save_indexed_dataset_infos(project_dir, indexed_dataset_infos)
    #     # <<<
    # indexed_dataset_infos = get_indexed_dataset_infos(project_dir)
    # <<<

    # Verify merged dbs.
    verify_merged_db(project_dir, indexed_dataset_infos, "sampled")
    verify_merged_db(project_dir, indexed_dataset_infos, "train")
    verify_merged_db(project_dir, indexed_dataset_infos, "valid")
