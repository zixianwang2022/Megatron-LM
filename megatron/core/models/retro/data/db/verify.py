# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
import torch
from typing import List, Tuple

from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.models.retro.data.config import RetroPreprocessingConfig
# from megatron.core.models.retro.data.external_libs import h5py
from megatron.core.models.retro.data.utils import (
    get_missing_blocks_by_rank,
    print_rank_0,
)

from .utils import (
    get_indexed_dataset_infos,
    get_merged_datasets,
)

# >>>
from lutil import pax
# <<<


def build_partial_db(
    config: RetroPreprocessingConfig,
    dataset_idx: int,
    n_datasets: int,
    indexed_dataset: MMapIndexedDataset,
    block_id: int,
    n_blocks: int,
    block: dict,
    proc_id: int,
    n_procs: int,
) -> Tuple[int, list, list, dict]:
    '''Process a document index range of the indexed dataset.

    The chunk database is built in parallel blocks, since de-tokenizing &
    re-tokenizing for Bert-length computation is expensive. This method
    iterates each document and extracts sequential 'chunk-length' sequences
    from each document.
    '''

    # Document start/end indexes.
    doc_range = block["range"]
    n_docs = doc_range[1] - doc_range[0]
    n_docs_per_proc = int(np.ceil(n_docs / n_procs))
    doc_start_id = doc_range[0] + proc_id * n_docs_per_proc
    doc_end_id = min(doc_range[1], doc_start_id + n_docs_per_proc)

    # Print progress.
    progress_proc_ids = set(range(n_procs)) \
        if torch.distributed.get_rank() == 0 else set()
    if proc_id in progress_proc_ids:
        print(" > building partial chunk db, proc %d / %d, docs %d:%d / %d."%(
            proc_id,
            n_procs,
            doc_start_id,
            doc_end_id,
            n_docs,
        ))

    # Progress bars (snapshot of overall progress).
    doc_id_iter = range(doc_start_id, doc_end_id)
    pbar = tqdm(doc_id_iter) \
        if proc_id in progress_proc_ids else \
           doc_id_iter

    # Iterate documents & parse chunks.
    chunk_db_valid = []
    chunk_db_invalid = []
    doc_size_map = {}
    for doc_id in pbar:

        # Progress description.
        try:
            pbar.set_description("ds %d / %d, block %d / %d, proc %d / %d." % (
                dataset_idx,
                n_datasets,
                block_id,
                n_blocks,
                proc_id,
                n_procs))
        except:
            pass

        # Remove EOD token.
        doc = indexed_dataset.get(doc_id)
        if doc[-1].item() == config.gpt_eod:
            doc = doc[:-1]
        doc_len = len(doc)

        # Chunk start/end indexes.
        chunk_start_idxs = list(range(0, doc_len, config.chunk_length))
        chunk_end_idxs = [min(doc_len, s + config.chunk_length)
                          for s in chunk_start_idxs]

        # Re-tokenize each chunk to Bert/Wordpiece (empty bert -> 'invalid').
        doc_size_map[doc_id] = 0
        for i, chunk_start_idx in enumerate(chunk_start_idxs):

            # Re-tokenize.
            chunk_end_idx = chunk_end_idxs[i]
            gpt_token_ids = indexed_dataset.get(
                idx=doc_id,
                offset=chunk_start_idx,
                length=chunk_end_idx - chunk_start_idx,
            )
            text = config.gpt_detokenize(gpt_token_ids.tolist())
            bert_token_ids = config.bert_tokenize(text)

            # 'Valid' for non-empty Bert chunks; 'invalid' otherwise.
            if len(bert_token_ids) == 0:
                _chunk_db = chunk_db_invalid
            else:
                _chunk_db = chunk_db_valid
                doc_size_map[doc_id] += 1
            _chunk_db.append((
                doc_id,
                chunk_start_idx,
                chunk_end_idx,
                len(bert_token_ids),
            ))

    return proc_id, chunk_db_valid, chunk_db_invalid, doc_size_map


def verify_individual_db(
    config: RetroPreprocessingConfig,
    dataset_idx: int,
    n_datasets: int,
    dataset_info: dict,
) -> None:
    '''Process a single indexed dataset & extract chunks.'''

    # >>>
    # # Make directory.
    # db_dir = dataset_info["db_dir"]
    # os.makedirs(db_dir, exist_ok=True)
    # <<<

    # Indexed dataset.
    indexed_dataset = dataset_info["dataset"]

    # >>>
    # # Missing db blocks.
    # n_missing_world, missing_db_blocks = get_missing_blocks_by_rank(
    #     db_dir,
    #     len(indexed_dataset),
    #     config.retro_doc_block_size,
    #     validate=lambda f : f["chunks_valid"].shape == (0,) \
    #         or f["chunks_valid"].shape[1] == 4)
    # <<<

    # >>>
    # # Prevent missing-path-write race condition.
    # torch.distributed.barrier()

    # if not missing_db_blocks:
    #     return
    # <<<

    # Num processes.
    # >>>
    # if n_missing_world == 1:
    #     n_procs = 128
    # elif n_missing_world <= 2:
    #     n_procs = 64
    # elif n_missing_world <= 4:
    #     n_procs = 32
    # elif n_missing_world <= 8:
    #     n_procs = 16
    # else:
    #     n_procs = 8
    # +++
    n_procs = 8
    # <<<

    # >>>
    raise Exception("hi.")
    # <<<

    # Process documents in parallel.
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for block_idx, block in enumerate(missing_db_blocks):

            if block is not None:

                db_path = block["path"]

                # Build partial dbs.
                print_rank_0(' > build partial dbs.')
                futures = []
                for proc_id in range(n_procs): # not true process id
                    futures.append(executor.submit(
                        build_partial_db,
                        types.SimpleNamespace(
                            chunk_length = config.retro_gpt_chunk_length,
                            gpt_eod = config.retro_tokenizers.gpt.eod,
                            gpt_detokenize = config.retro_tokenizers.gpt.detokenize,
                            bert_tokenize = config.retro_tokenizers.bert.tokenize,
                        ),
                        dataset_idx,
                        n_datasets,
                        indexed_dataset,
                        block_idx,
                        len(missing_db_blocks),
                        block,
                        proc_id,
                        n_procs,
                    ))
                partial_chunk_dbs = []
                for future in as_completed(futures):
                    partial_chunk_dbs.append(future.result())

                # Concatenate chunks.
                partial_chunk_dbs.sort(key=lambda item:item[0]) # sort by proc_id
                chunk_db_valid = [item
                                  for partial_chunk_db in partial_chunk_dbs
                                  for item in partial_chunk_db[1]]
                chunk_db_invalid = [item
                                    for partial_chunk_db in partial_chunk_dbs
                                    for item in partial_chunk_db[2]]

                # Convert to numpy.
                print_rank_0(' > converting chunk db to numpy.')
                chunk_db_valid = np.array(chunk_db_valid, dtype="uint32")
                chunk_db_invalid = np.array(chunk_db_invalid, dtype="uint32")

                # Document offsets.
                doc_sizes = [(d, s)
                             for partial_chunk_db in partial_chunk_dbs
                             for d, s in partial_chunk_db[3].items()]
                doc_sizes.sort(key = lambda item : item[0])
                doc_offsets = np.cumsum([item[1] for item in doc_sizes]) \
                                .astype("uint64")
                doc_offsets = np.stack((
                    np.array([item[0] for item in doc_sizes], dtype="uint64"),
                    doc_offsets), axis=1)

                # Save DB.
                print_rank_0(" > saving individual db.")
                with h5py.File(db_path, "w") as f:
                    dset = f.create_dataset("chunks_valid", data=chunk_db_valid)
                    dset = f.create_dataset("chunks_invalid",
                                            data=chunk_db_invalid)
                    dset = f.create_dataset("doc_offsets", data=doc_offsets)

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
