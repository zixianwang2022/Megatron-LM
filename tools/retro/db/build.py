# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import reduce
import glob
import json
import numpy as np
import os
from pathlib import Path
# >>>
import sqlite3
# <<<
import threading
import torch
from tqdm import tqdm
import types

from megatron import get_retro_args, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)
from tools.bert_embedding.utils import get_missing_blocks_by_rank
from tools.retro.external_libs import h5py
from tools.retro.utils import get_gpt_tokenizer, get_bert_tokenizer

from .utils import (
    get_banned_doc_hash,
    get_indexed_dataset_infos,
    get_indexed_dataset_infos_path,
    get_individual_db,
    get_individual_db_dir,
    get_merged_dataset,
    get_merged_db_path_map,
    get_train_banned_doc_db_path,
    get_train_banned_doc_json_dir,
    save_indexed_dataset_infos,
)

# >>>
import time
from lutil import pax
# <<<


def init_indexed_dataset_infos():
    '''Gather meta-info about each indexed dataset.

    The returned info array allows for easy access to the configuration, and
    helps remove ambiguity.
    '''

    args = get_retro_args()

    assert len(args.data_path) % 2 == 0, \
        "currently, only blendable dataset is supported."

    # Dataset infos.
    infos = []
    for i in range(0, len(args.data_path), 2):
        ratio = float(args.data_path[i])
        prefix = args.data_path[i + 1]
        path = prefix + ".bin"
        name = os.path.basename(prefix)
        assert os.path.exists(path), "couldn't find '%s'." % path
        infos.append({
            "ratio" : ratio,
            "prefix" : prefix,
            "path" : path,
            "name" : name,
            "db_dir" : get_individual_db_dir(name),
            "dataset" : make_indexed_dataset(prefix, "mmap", True),
        })

    # >>>
    # info = infos[0]
    # pax({"info": info, "dataset": len(info["dataset"])})
    # <<<

    return infos


def build_partial_db(
        dataset_idx,
        n_datasets,
        indexed_dataset,
        block_id,
        n_blocks,
        block,
        proc_id,
        n_procs,
        tokenizers,
):
    '''Process a document index range of the indexed dataset.

    The chunk database is built in parallel blocks, since de-tokenizing &
    re-tokenizing for Bert-length computation is expensive. This method
    iterates each document and extracts sequential 'chunk-length' sequences
    from each document.
    '''

    args = get_retro_args()

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
        if doc[-1].item() == tokenizers.gpt.eod:
            doc = doc[:-1]
        doc_len = len(doc)

        # Chunk start/end indexes.
        chunk_start_idxs = list(range(0, doc_len, args.retro_gpt_chunk_length))
        chunk_end_idxs = [min(doc_len, s + args.retro_gpt_chunk_length)
                          for s in chunk_start_idxs]

        # Re-tokenize each chunk to Bert/Wordpiece (empty bert -> 'invalid').
        for i, chunk_start_idx in enumerate(chunk_start_idxs):

            # Re-tokenize.
            chunk_end_idx = chunk_end_idxs[i]
            gpt_token_ids = indexed_dataset.get(
                idx=doc_id,
                offset=chunk_start_idx,
                length=chunk_end_idx - chunk_start_idx,
            )
            text = tokenizers.gpt.detokenize(gpt_token_ids.tolist())
            bert_token_ids = tokenizers.bert.tokenize(text)

            # 'Valid' for non-empty Bert chunks; 'invalid' otherwise.
            _chunk_db = chunk_db_invalid \
                if len(bert_token_ids) == 0 else \
                   chunk_db_valid
            _chunk_db.append((
                doc_id,
                chunk_start_idx,
                chunk_end_idx,
                len(bert_token_ids),
            ))

    return proc_id, chunk_db_valid, chunk_db_invalid


def build_individual_db(dataset_idx, n_datasets, dataset_info, tokenizers):
    '''Process a single indexed dataset & extract chunks.'''

    args = get_retro_args()

    # Make directory.
    db_dir = dataset_info["db_dir"]
    os.makedirs(db_dir, exist_ok=True)

    # Indexed dataset.
    indexed_dataset = dataset_info["dataset"]

    # Missing db blocks.
    n_missing_world, missing_db_blocks = get_missing_blocks_by_rank(
        db_dir,
        len(indexed_dataset),
        args.retro_doc_block_size,
        validate=lambda f : f["chunks_valid"].shape == (0,) \
            or f["chunks_valid"].shape[1] == 4)

    # Prevent missing-path-write race condition.
    torch.distributed.barrier()

    if not missing_db_blocks:
        return

    # Num processes.
    if n_missing_world == 1:
        n_procs = 128
    elif n_missing_world <= 2:
        n_procs = 64
    elif n_missing_world <= 4:
        n_procs = 32
    elif n_missing_world <= 8:
        n_procs = 16
    else:
        n_procs = 8

    # Process documents in parallel.
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for block_idx, block in enumerate(missing_db_blocks):

            if block is not None:

                # Build partial dbs.
                print_rank_0(' > build partial dbs.')
                futures = []
                for proc_id in range(n_procs): # not true process id
                    futures.append(executor.submit(
                        build_partial_db,
                        dataset_idx,
                        n_datasets,
                        indexed_dataset,
                        block_idx,
                        len(missing_db_blocks),
                        block,
                        proc_id,
                        n_procs,
                        tokenizers,
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
                chunk_db_valid = np.array(chunk_db_valid)
                chunk_db_invalid = np.array(chunk_db_invalid)

                # Save DB.
                print_rank_0(" > saving individual db.")
                f = h5py.File(block["path"], "w")
                dset = f.create_dataset("chunks_valid", data=chunk_db_valid)
                dset = f.create_dataset("chunks_invalid", data=chunk_db_invalid)
                f.close()

            # Wait for all ranks to finish block.
            print_rank_0(" > waiting for all ranks to finish block.")
            torch.distributed.barrier()

    print_rank_0(" > finished saving individual db.")


def build_individual_dbs(indexed_dataset_infos):
    '''Iterate each indexed dataset & process its chunks.'''

    args = get_retro_args()

    # Tokenizers.
    tokenizers = types.SimpleNamespace(
        gpt=get_gpt_tokenizer(),
        bert=get_bert_tokenizer(),
    )

    # Build individual DBs.
    print_rank_0(" > build individual chunk dbs.")
    for ds_idx, ds_info in enumerate(indexed_dataset_infos):

        # Progress.
        print_rank_0(" > building individual db, dataset %d / %d ... '%s'." % (
            ds_idx,
            len(indexed_dataset_infos),
            ds_info["name"],
        ))

        # Process single dataset.
        build_individual_db(ds_idx, len(indexed_dataset_infos),
                            ds_info, tokenizers)


def update_chunk_counts(indexed_dataset_infos):
    '''Set n_chunks_train & n_chunks sampled for each individual DB.'''

    args = get_retro_args()

    if torch.distributed.get_rank() != 0:
        return

    # Data ratio sum (for setting index training chunks).
    data_ratio_sum = sum([ d["ratio"] for d in indexed_dataset_infos ])

    # Training split size (split at document level).
    train_fraction = float(args.split.split(",")[0]) / 100
    assert train_fraction > 0 and train_fraction <= 1

    # Set n_chunks (including n_chunks_sampled for unambiguity).
    print_rank_0(" > compute n_chunks.")
    for ds_index, ds_info in enumerate(indexed_dataset_infos):

        db_dir = ds_info["db_dir"]
        db_paths = sorted(glob.glob(db_dir + "/*.hdf5"))

        # Update counts.
        ds_info["n_docs"] = len(ds_info["dataset"].doc_idx) - 1
        ds_info["n_docs_train"] = int(train_fraction * ds_info["n_docs"])
        ds_info["n_chunks"] = 0 # previously, 'n_chunks_valid'
        ds_info["n_chunks_train"] = 0
        ds_info["n_chunks_invalid"] = 0
        for db_path in tqdm(db_paths, "%d/%d, %s" % (
                ds_index, len(indexed_dataset_infos), ds_info["name"])):
           with h5py.File(db_path, "r") as f:
                ds_info["n_chunks"] += len(f["chunks_valid"])
                ds_info["n_chunks_invalid"] += len(f["chunks_invalid"])
                ds_info["n_chunks_train"] += \
                    (np.copy(f["chunks_valid"][:, 0]) < ds_info["n_docs_train"]) \
                    .sum().item()

        ds_info["n_chunks_sampled"] = int(args.retro_nchunks_sampled *
                                          ds_info["ratio"] / data_ratio_sum)

        # Verify counts.
        assert ds_info["n_chunks_train"] <= ds_info["n_chunks"], \
            "n_train (%d) > n_total (%d)." % (
                ds_info["n_chunks_train"], ds_info["n_chunks"])
        assert ds_info["n_chunks_sampled"] <= ds_info["n_chunks_train"], \
            "n_sampled (%d) > n_train (%d)." % (
                ds_info["n_chunks_sampled"], ds_info["n_chunks_train"])


def merge_dbs(indexed_dataset_infos, db_type):
    '''Merge individual DBs into single DB.'''

    if torch.distributed.get_rank() != 0:
        return

    print(" > build %s chunk db." % db_type)

    # Count chunks.
    if db_type == "full":
        raise Exception("deprecated; use 'train' or 'sampled'.")
        n_chunks_key = "n_chunks"
    elif db_type == "sampled":
        n_chunks_key = "n_chunks_sampled"
    elif db_type == "train":
        n_chunks_key = "n_chunks_train"
    elif db_type == "valid":
        pass
    else:
        raise Exception("handle db_type '%s'." % db_type)

    if db_type == "valid":
        n_chunks = sum(m["n_chunks"] - m["n_chunks_train"]
                       for m in indexed_dataset_infos)
    else:
        n_chunks = sum(m[n_chunks_key] for m in indexed_dataset_infos)

    # DB path.
    db_path = get_merged_db_path_map()[db_type]

    # Delete existing chunk db if incorrect size.
    if os.path.exists(db_path):

        try:

            f = h5py.File(db_path)
            n_alloc = len(f["chunks"])           # total allocated
            n_written = f["n_written"][0].item() # total written
            f.close()

            if n_chunks != n_alloc or n_chunks != n_written:
                os.remove(db_path)

        except Exception as e:
            if isinstance(e, OSError):
                os.remove(db_path)
            elif isinstance(e, KeyError):
                f.close()
                os.remove(db_path)
            else:
                raise e

    # Build merged chunk db.
    if not os.path.exists(db_path):

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        f = h5py.File(db_path, "w")

        # Initialize output arrays.
        merged_db = f.create_dataset("chunks", (n_chunks, 5), dtype="i8")
        n_written = f.create_dataset("n_written", (1,), dtype="uint64")
        n_written[0] = 0

        # Iterate indexed datasets & collect chunks.
        start_index = 0
        for ds_idx, ds_info in enumerate(indexed_dataset_infos):
            print(" > merging dbs; '%s', dataset %d / %d ... '%s'." %
                  (db_type, ds_idx, len(indexed_dataset_infos), ds_info["name"]))
            individual_db = get_individual_db(ds_idx, ds_info)

            if db_type == "valid":
                individual_db = individual_db[ds_info["n_chunks_train"]:]
            else:
                individual_db = individual_db[:ds_info[n_chunks_key]]

            merged_db[start_index:start_index+len(individual_db)] = individual_db
            start_index += len(individual_db)
            n_written[0] = start_index

        f.close()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def get_partial_banned_chunk_map(proc_id, db_path, chunk_range_info):
#     '''Build partial mapping of {(dataset_id,doc_id):[chunk_ids]}.

#     In this method, only chunks within the range (start_chunk_id, end_chunk_id]
#     are processed.'''

#     start_chunk_id = chunk_range_info["start"]
#     end_chunk_id = chunk_range_info["end"]
#     output_path = chunk_range_info["path"]

#     # Skip, if output file exists.
#     if os.path.exists(output_path):
#         return

#     # Chunk subset.
#     with h5py.File(db_path) as f:
#         sub_chunk_db = np.copy(f["chunks"][start_chunk_id:end_chunk_id, :2])

#     # Map docs to chunks.
#     banned_chunk_map = defaultdict(list)
#     for rel_chunk_id, (dataset_id, doc_id) in enumerate(tqdm(
#             sub_chunk_db,
#             "map banned docs, proc %d" % proc_id,
#             total=sub_chunk_db.shape[0],
#     )):
#         chunk_id = start_chunk_id + rel_chunk_id
#         banned_chunk_map["%d,%d" % (dataset_id.item(), doc_id.item())] \
#             .append(chunk_id)

#     # Save output.
#     # >>>
#     with open(output_path, "w") as f:
#         json.dump(banned_chunk_map, f)
#     # +++
#     # pax({
#     #     # "banned_chunk_map" : banned_chunk_map,
#     #     "banned_chunk_map / 0" : list(banned_chunk_map.items())[0],
#     # })
#     # <<<


# # >>>
# # def get_train_doc_chunk_map():
# def merge_doc_chunk_maps():
#     '''Merge multiple doc map jsons into sqlite database.'''

#     # Connect to database.
#     db_path = get_train_banned_doc_db_path()
#     with sqlite3.connect(db_path) as conn:

#         conn.row_factory = sqlite3.Row
#         cursor = conn.cursor()

#         # Init tables.
#         rs = cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")
#         table_names = set(r["name"] for r in rs)
#         # if "doc_chunks" not in table_names:
#         if not table_names:
#             cursor.execute("CREATE TABLE doc_chunks ("
#                            "  doc_hash INTEGER PRIMARY KEY,"
#                            "  doc_key TEXT NOT NULL,"
#                            "  chunk_ids TEXT NOT NULL"
#                            ")")
#             cursor.execute("CREATE TABLE completed_paths (path TEXT NOT NULL)")
#             # cursor.execute("CREATE TABLE completed (completed INTEGER)")

#         # Individual json map paths.
#         completed_paths = cursor.execute("SELECT * FROM completed_paths")
#         completed_paths = set(r["path"] for r in completed_paths)
#         paths = sorted(glob.glob(get_train_banned_doc_json_dir() + "/*.json"))
#         paths = [ p for p in paths if os.path.basename(p) not in completed_paths ]

#         # pax({"completed_paths": completed_paths, "paths": paths[:10]})

#         # Iterate json map paths.
#         doc_map = defaultdict(set)
#         # for path_index, path in enumerate(tqdm(paths, "merge train doc maps")):
#         for path_index, path in enumerate(paths):

#             # Loaded doc map.
#             # >>>
#             # with open(path) as f:
#             #     loaded_doc_map = json.load(f)
#             #     loaded_doc_map = {
#             #         int(hashlib.sha256(doc_key.encode()).hexdigest()[:10], 16):(
#             #             tuple(int(i) for i in doc_key.split(",")),
#             #             set(chunk_ids),
#             #         ) for doc_key, chunk_ids in loaded_doc_map.items()}
#             # +++
#             with open(path) as f:
#                 loaded_doc_map = {}
#                 _loaded_doc_map = json.load(f)
#                 for doc_tuple_str, chunk_id_str in _loaded_doc_map.items():
#                     doc_tuple = tuple(int(i) for i in doc_tuple_str.split(","))
#                     doc_hash = get_banned_doc_hash(*doc_tuple)
#                     chunk_ids = set(chunk_id_str)
#                     # pax({
#                     #     "doc_tuple" : doc_tuple,
#                     #     "doc_hash" : doc_hash,
#                     #     "chunk_ids" : chunk_ids,
#                     # })
#                     loaded_doc_map[doc_hash] = (doc_tuple, chunk_ids)
#             # <<<

#             # Existing doc map.
#             existing_rows = cursor.execute("SELECT * FROM doc_chunks WHERE doc_hash IN (%s)" % ",".join(str(i) for i in loaded_doc_map.keys()))
#             existing_doc_map = {r["doc_hash"]: (
#                 tuple(json.loads(r["doc_key"])),
#                 set(json.loads(r["chunk_ids"])),
#             ) for r in existing_rows}

#             # Add to doc map.
#             merged_doc_map = existing_doc_map
#             for doc_hash, (doc_key, loaded_chunk_ids) in loaded_doc_map.items():
#                 existing_entry = existing_doc_map.get(doc_hash, (None, set()))
#                 existing_chunk_ids = existing_entry[1]
#                 merged_chunk_ids = existing_chunk_ids | loaded_chunk_ids
#                 assert len(merged_chunk_ids) == \
#                     len(loaded_chunk_ids) + len(existing_chunk_ids)
#                 merged_doc_map[doc_hash] = (doc_key, merged_chunk_ids)

#             # Insert into database.
#             insert_rows = [
#                 (doc_hash, json.dumps(list(doc_key)), json.dumps(list(chunk_ids)))
#                 for doc_hash, (doc_key, chunk_ids) in merged_doc_map.items()]
#             # >>>
#             # pax({
#             #     "insert_rows" : len(insert_rows),
#             #     "insert_rows / 0" : insert_rows[:10],
#             # })
#             # <<<
#             block_size = 10000
#             pbar = tqdm(range(0, len(insert_rows), block_size))
#             for row_start in pbar:
#                 pbar.set_description("merge banned docs %d / %d" %
#                                      (path_index, len(paths)))
#                 row_end = min(len(insert_rows), row_start + block_size)
#                 block_rows = insert_rows[row_start:row_end]
#                 cursor.execute("INSERT OR REPLACE INTO doc_chunks (doc_hash, doc_key, chunk_ids) VALUES %s" % ",".join("(?,?,?)" for _ in range(len(block_rows))), [item for row in block_rows for item in row ])
#                 conn.commit()

#             cursor.execute("INSERT INTO completed_paths (path) VALUES (?)", (os.path.basename(path),))
#             conn.commit()
# # <<<


# # >>>
# # def build_doc_chunk_map(indexed_dataset_infos, db_type):
# def build_banned_doc_db(indexed_dataset_infos, db_type):
# # <<<
#     '''Build mapping of {(dataset_id,doc_id):[chunk_ids]}.'''

#     if torch.distributed.get_rank() != 0:
#         return

#     print(" > build %s doc-chunk map." % db_type)

#     n_procs = 128

#     # Get dataset.
#     db_dataset = get_merged_dataset(db_type, indexed_dataset_infos)

#     # Sub-ranges for parallel processing.
#     n_chunks = db_dataset.chunks.shape[0]
#     n_chunks_per_proc = max(1, int(np.ceil(n_chunks / n_procs)))
#     chunk_id_starts = list(range(0, n_chunks, n_chunks_per_proc))
#     chunk_id_ranges = [(s, min(n_chunks, s + n_chunks_per_proc))
#                        for s in chunk_id_starts]

#     # Wrap range info with output path.
#     n_digits = int(np.ceil(np.log(n_chunks) / np.log(10)) + 1)
#     output_dirname = get_train_banned_doc_json_dir()
#     chunk_range_infos = [{
#         "start" : start_id,
#         "end" : end_id,
#         "path" : os.path.join(output_dirname, "%s-%s.json" % (
#             str(start_id).zfill(n_digits),
#             str(end_id).zfill(n_digits),
#         )),
#     } for start_id, end_id in chunk_id_ranges ]

#     # Build doc-chunk map.
#     print_rank_0("build doc-chunk-map.")
#     with ProcessPoolExecutor(max_workers=n_procs) as executor:

#         # Build partial chunk maps.
#         futures = []
#         for proc_id, chunk_range_info in enumerate(chunk_range_infos):

#             if os.path.exists(chunk_range_info["path"]):
#                 continue

#             # Submit job.
#             futures.append(executor.submit(
#                 get_partial_banned_chunk_map,
#                 proc_id,
#                 db_dataset.db_path,
#                 chunk_range_info,
#             ))

#         # Wait for processes to finish.
#         banned_chunk_paths = []
#         for finished_idx, future in enumerate(as_completed(futures)):
#             print("finished %d / %d." % (finished_idx, n_procs))
#             future.result()

#     # >>>
#     # Merge json maps into sqlite db.
#     merge_doc_chunk_maps()
#     # <<<
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # >>>
# def convert_dataset_to_sqlite(db_dataset):

#     if torch.distributed.get_rank() != 0:
#         return

#     print(" > convert dataset -> sqlite.")

#     db_path = get_train_banned_doc_db_path()
#     db_path = os.path.join(os.path.dirname(db_path), "tmp_chunks.db")

#     # pax({
#     #     "db_dataset" : db_dataset,
#     #     "db_path" : db_path,
#     # })

#     conn = sqlite3.connect(db_path)
#     conn.row_factory = sqlite3.Row
#     cursor = conn.cursor()

#     # Init tables.
#     rs = cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")
#     table_names = set(r["name"] for r in rs)
#     if not table_names:
#         cursor.execute("CREATE TABLE chunks ("
#                        "  dataset INTEGER NOT NULL,"
#                        "  document INTEGER NOT NULL,"
#                        "  token_start INTEGER NOT NULL,"
#                        "  token_end INTEGER NOT NULL,"
#                        "  bert_len INTEGER NOT NULL"
#                        ")")
#         cursor.execute("CREATE TABLE n_chunks_completed (n INTEGER NOT NULL)")

#     rs = cursor.execute("SELECT n FROM n_chunks_completed")
#     n_chunks_completed = sorted([r["n"] for r in rs])
#     n_chunks_completed = n_chunks_completed[-1] if n_chunks_completed else 0

#     block_size = 10000
#     for chunk_start_idx in tqdm(
#             range(n_chunks_completed, len(db_dataset), block_size),
#             "converties"):

#         chunk_end_idx = min(len(db_dataset), chunk_start_idx + block_size)

#         rows = db_dataset.chunks[chunk_start_idx:chunk_end_idx]
#         rows = [ tuple(r.tolist()) for r in rows ]
#         cursor.execute("INSERT INTO chunks VALUES %s" % ",".join(["(?,?,?,?,?)"]*len(rows)), [item for row in rows for item in row])
#         cursor.execute("INSERT INTO n_chunks_completed (n) VALUES (?)",
#                        (chunk_end_idx,))
#         conn.commit()

#     raise Exception("hi.")
# # <<<

# def build_banned_doc_db(indexed_dataset_infos, db_type):
#     '''Build mapping of {(dataset_id,doc_id):[chunk_ids]}.'''

#     if torch.distributed.get_rank() != 0:
#         return

#     print(" > build %s banned doc map." % db_type)

#     # Get dataset.
#     db_dataset = get_merged_dataset(db_type, indexed_dataset_infos)

#     # >>>
#     convert_dataset_to_sqlite(db_dataset)
#     raise Exception("hi.")
#     # <<<

#     # pax({
#     #     "indexed_dataset_infos": indexed_dataset_infos,
#     #     "db_dataset" : db_dataset,
#     #     "db_dataset / len" : len(db_dataset),
#     # })

#     # Connect to database.
#     db_path = get_train_banned_doc_db_path()
#     conn = sqlite3.connect(db_path)
#     conn.row_factory = sqlite3.Row
#     cursor = conn.cursor()

#     # Init tables.
#     rs = cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")
#     table_names = set(r["name"] for r in rs)
#     if not table_names:
#         cursor.execute("CREATE TABLE doc_chunks ("
#                        "  doc_hash INTEGER PRIMARY KEY,"
#                        "  doc_tuple TEXT NOT NULL,"
#                        "  chunk_ids TEXT NOT NULL"
#                        ")")
#         cursor.execute("CREATE TABLE n_chunks_completed (n INTEGER NOT NULL)")

#     # >>>
#     # print("insert.")
#     # conn.execute("INSERT INTO doc_chunks (doc_hash, doc_tuple, chunk_ids) VALUES (?, ?, ?)", (0, "hi", "there"))
#     # print("inserted.")
#     # conn.commit()
#     # print("committed.")
#     # conn.close()
#     # print("closed.")
#     # exit()
#     # <<<

#     rs = cursor.execute("SELECT n FROM n_chunks_completed")
#     n_chunks_completed = sorted([r["n"] for r in rs])
#     n_chunks_completed = n_chunks_completed[-1] if n_chunks_completed else 0
#     # if n_chunks_completed not in (100000,):
#     #     pax({"n_chunks_completed": n_chunks_completed})

#     time_map = {}
#     block_size = 1000000
#     # block_size = 10000000
#     # for chunk_start_idx in tqdm(
#     #         range(n_chunks_completed, len(db_dataset), block_size),
#     #         "banned docs"):
#     pbar = tqdm(range(n_chunks_completed, len(db_dataset), block_size))
#     for chunk_start_idx in pbar:

#         chunk_end_idx = min(len(db_dataset), chunk_start_idx + block_size)

#         # Current doc map.
#         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#         t = time.time()
#         current_doc_map = defaultdict(set)
#         for local_chunk_idx, doc_entry in \
#             enumerate(db_dataset.chunks[chunk_start_idx:chunk_end_idx, :2]):
#             doc_tuple = tuple(doc_entry.tolist())
#             current_doc_map[doc_tuple].add(chunk_start_idx + local_chunk_idx)
#         time_map["current / map"] = time.time() - t
#         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         # t = time.time()
#         # current_entries = \
#         #     db_dataset.chunks[chunk_start_idx:chunk_end_idx, :2].tolist()
#         # time_map["current / entries"] = time.time() - t

#         # t = time.time()
#         # current_tuples = [ tuple(e) for e in current_entries ]
#         # time_map["current / tuples"] = time.time() - t

#         # t = time.time()
#         # current_doc_map = defaultdict(set)
#         # # for local_chunk_idx, doc_entry in enumerate(current_entries):
#         # #     doc_tuple = tuple(doc_entry.tolist())
#         # #     current_doc_map[doc_tuple].add(chunk_start_idx + local_chunk_idx)
#         # for local_chunk_idx, doc_tuple in enumerate(current_tuples):
#         #     current_doc_map[doc_tuple].add(chunk_start_idx + local_chunk_idx)
#         # time_map["current / map"] = time.time() - t
#         # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#         t = time.time()
#         current_doc_map = {get_banned_doc_hash(*t) : (t, c)
#                            for t, c in current_doc_map.items()}
#         time_map["current / hash"] = time.time() - t
#         # pax({"%d / %s" % (i, t) : ("%d / %s" % (len(c), str(c)))
#         #      for i,(t,c) in current_doc_map.items()})

#         # Existing doc map.
#         t = time.time()
#         existing_rows = cursor.execute("SELECT * FROM doc_chunks WHERE doc_hash IN (%s)" % ",".join(str(i) for i in current_doc_map.keys()))
#         existing_doc_map = {r["doc_hash"]: (
#             tuple(json.loads(r["doc_tuple"])),
#             set(json.loads(r["chunk_ids"])),
#         ) for r in existing_rows}
#         time_map["existing"] = time.time() - t

#         # Add to doc map.
#         t = time.time()
#         merged_doc_map = {}
#         for doc_hash,(doc_tuple,current_chunk_ids) in current_doc_map.items():
#             existing_entry = existing_doc_map.get(doc_hash, (None, set()))
#             existing_chunk_ids = existing_entry[1]
#             merged_chunk_ids = existing_chunk_ids | current_chunk_ids
#             assert len(merged_chunk_ids) == \
#                 len(current_chunk_ids) + len(existing_chunk_ids)
#             merged_doc_map[doc_hash] = (doc_tuple, merged_chunk_ids)
#         time_map["merge"] = time.time() - t

#         # >>>
#         # if existing_doc_map:
#         #     pax({
#         #         "chunk range" : str((chunk_start_idx, chunk_end_idx)),
#         #         "current_doc_map" : len(current_doc_map),
#         #         "existing_doc_map" : len(existing_doc_map),
#         #         "merged_doc_map" : len(merged_doc_map),
#         #     })
#         #     pax({str(i):"%s | %d | %s" % (t, len(c), c) for i, (t, c) in merged_doc_map.items()})
#         # <<<

#         # Insert into database.
#         t = time.time()
#         insert_rows = [(
#             doc_hash,
#             json.dumps(list(doc_tuple)),
#             json.dumps(list(chunk_ids)),
#         ) for doc_hash, (doc_tuple, chunk_ids) in merged_doc_map.items()]
#         time_map["insert rows"] = time.time() - t

#         t = time.time()
#         cursor.execute("INSERT OR REPLACE INTO doc_chunks (doc_hash, doc_tuple, chunk_ids) VALUES %s" % ",".join("(?,?,?)" for _ in range(len(insert_rows))), [item for row in insert_rows for item in row ])
#         cursor.execute("INSERT INTO n_chunks_completed (n) VALUES (?)",
#                        (chunk_end_idx,))
#         time_map["insert"] = time.time() - t

#         t = time.time()
#         conn.commit()
#         time_map["commit"] = time.time() - t

#         pbar.set_description("banned docs, ds %d, rows %d" % (
#             list(merged_doc_map.values())[0][0][0],
#             len(insert_rows)))

#         pax({
#             "time_map" : time_map,
#             "insert_rows" : [ "%s, %d" % (r[1], len(json.loads(r[2])))
#                               for r in insert_rows[:100] ],
#             "insert_rows / len" : len(insert_rows),
#         })

#         # >>>
#         # print("committed; exit.")
#         # conn.close()
#         # raise Exception("hi.")
#         # <<<
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def build_banned_doc_db(indexed_dataset_infos, db_type):
    '''Build mapping of {(dataset_id,doc_id):[chunk_ids]}.'''

    if torch.distributed.get_rank() != 0:
        return

    print(" > build %s banned doc map." % db_type)

    # Get dataset.
    db_dataset = get_merged_dataset(db_type, indexed_dataset_infos)

    # Connect to database.
    db_path = get_train_banned_doc_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Init tables.
    rs = cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")
    table_names = set(r["name"] for r in rs)
    if not table_names:
        cursor.execute("CREATE TABLE doc_chunks ("
                       "  doc_hash INTEGER PRIMARY KEY,"
                       "  dataset INTEGER NOT NULL,"
                       "  doc INTEGER NOT NULL,"
                       "  chunk_min INTEGER NOT NULL,"
                       "  chunk_max INTEGER NOT NULL"
                       ")")
        cursor.execute("CREATE TABLE n_chunks_completed (n INTEGER NOT NULL)")

    rs = cursor.execute("SELECT n FROM n_chunks_completed")
    n_chunks_completed = sorted([r["n"] for r in rs])
    n_chunks_completed = n_chunks_completed[-1] if n_chunks_completed else 0

    time_map = {}
    block_size = 100000
    pbar = tqdm(range(n_chunks_completed, len(db_dataset), block_size))
    for start_chunk_id in pbar:
        end_chunk_id = min(len(db_dataset), start_chunk_id + block_size)

        # Current doc map.
        t = time.time()
        current_doc_map = {} # defaultdict(set)
        for local_chunk_id, doc_entry in \
            enumerate(db_dataset.chunks[start_chunk_id:end_chunk_id, :2]):
            doc_tuple = tuple(doc_entry.tolist())
            chunk_id = start_chunk_id + local_chunk_id
            current_range = current_doc_map.get(doc_tuple, [chunk_id, chunk_id])
            current_range[0] = min(current_range[0], chunk_id)
            current_range[1] = max(current_range[1], chunk_id)
            current_doc_map[doc_tuple] = current_range
        time_map["current / map"] = time.time() - t

        # pax({
        #     "current_doc_map" : {k : "%d / %s" % (v[1]-v[0], v)
        #                          for k, v in current_doc_map.items()},
        #     "current_doc_map" : len(current_doc_map),
        # })

        t = time.time()
        current_doc_map = {get_banned_doc_hash(*t) : (*t, *c)
                           for t, c in current_doc_map.items()}
        time_map["current / hash"] = time.time() - t

        # pax({"current_doc_map": current_doc_map})

        # Existing doc map.
        t = time.time()
        existing_rows = cursor.execute("SELECT * FROM doc_chunks WHERE doc_hash IN (%s)" % ",".join(str(i) for i in current_doc_map.keys()))
        existing_doc_map = {r["doc_hash"]: (
            r["dataset"],
            r["doc"],
            r["chunk_min"],
            r["chunk_max"],
        ) for r in existing_rows}
        time_map["existing"] = time.time() - t

        # if existing_doc_map:
        #     pax({"existing_doc_map": existing_doc_map})

        # Add to doc map.
        t = time.time()
        merged_doc_map = {}
        for doc_hash, (dataset_id, doc_id, min_chunk_id, max_chunk_id) \
            in current_doc_map.items():

            # pax({
            #     "doc_hash" : doc_hash,
            #     "dataset_id" : dataset_id,
            #     "doc_id" : doc_id,
            #     "min_chunk_id" : min_chunk_id,
            #     "max_chunk_id" : max_chunk_id,
            # })

            existing_entry = existing_doc_map.get(doc_hash, None)
            if existing_entry:
                min_chunk_id = min(min_chunk_id, existing_entry[2])
                max_chunk_id = max(max_chunk_id, existing_entry[3])
                # pax({
                #     "existing_entry" : existing_entry,
                #     "new entry" : (None, None, min_chunk_id, max_chunk_id),
                # })
            merged_doc_map[doc_hash] = \
                (dataset_id, doc_id, min_chunk_id, max_chunk_id)
        time_map["merge"] = time.time() - t

        # >>>
        # if existing_doc_map:
        #     common_doc_hashes = \
        #         set(current_doc_map.keys()) & set(existing_doc_map.keys())
        #     pax({
        #         "chunk range" : str((start_chunk_id, end_chunk_id)),
        #         # "current_doc_map" : len(current_doc_map),
        #         # "existing_doc_map" : len(existing_doc_map),
        #         # "merged_doc_map" : len(merged_doc_map),
        #         "common_doc_hashes" : common_doc_hashes,
        #         "current doc map" :
        #         {h:current_doc_map[h] for h in common_doc_hashes},
        #         "existing doc map" :
        #         {h:existing_doc_map[h] for h in common_doc_hashes},
        #         "merged doc map" :
        #         {h:merged_doc_map[h] for h in common_doc_hashes},
        #     })
        #     pax({str(i):"%s | %d | %s" % (t, len(c), c) for i, (t, c) in merged_doc_map.items()})
        # <<<

        # Insert into database.
        t = time.time()
        insert_rows = [(
            doc_hash,
            *entry,
        ) for doc_hash, entry in merged_doc_map.items()]
        time_map["insert rows"] = time.time() - t

        # pax({"insert_rows": insert_rows})

        t = time.time()
        cursor.execute("INSERT OR REPLACE INTO doc_chunks VALUES %s" % ",".join("(?,?,?,?,?)" for _ in range(len(insert_rows))), [item for row in insert_rows for item in row ])
        cursor.execute("INSERT INTO n_chunks_completed (n) VALUES (?)",
                       (end_chunk_id,))
        time_map["insert"] = time.time() - t

        t = time.time()
        conn.commit()
        time_map["commit"] = time.time() - t

        pbar.set_description("banned docs, ds %d, rows %d" % (
            list(merged_doc_map.values())[0][0],
            len(insert_rows)))

        # pax({
        #     "time_map" : time_map,
        #     "insert_rows" : [ "%s, %d" % (r[1], len(json.loads(r[2])))
        #                       for r in insert_rows[:100] ],
        #     "insert_rows / len" : len(insert_rows),
        # })

        # >>>
        # print("committed; exit.")
        # conn.close()
        # raise Exception("hi.")
        # <<<
# <<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def build_db():
    '''Extract token chunks from each indexed dataset.

    Iterate each document of each indexed dataset, extract that document's
    chunks, and save to a 'DB' (hdf5 file).
    '''

    # Indexed dataset info.
    indexed_dataset_infos = init_indexed_dataset_infos()

    # >>>
    if 1:
    # <<<
        # Build dbs.
        build_individual_dbs(indexed_dataset_infos)

        # Single-process going forward.
        if torch.distributed.get_rank() != 0:
            return

    # Update n_chunks & save indexed dataset infos.
    if not os.path.exists(get_indexed_dataset_infos_path()):
        update_chunk_counts(indexed_dataset_infos)
        save_indexed_dataset_infos(indexed_dataset_infos)
    indexed_dataset_infos = get_indexed_dataset_infos()

    # >>>
    # pax({"indexed_dataset_infos": indexed_dataset_infos})
    # <<<

    # Merge dbs.
    # >>>
    if 1:
    # <<<
        merge_dbs(indexed_dataset_infos, "sampled")
        merge_dbs(indexed_dataset_infos, "train")
        merge_dbs(indexed_dataset_infos, "valid")

    # >>>
    # # Build banned document map.
    # build_banned_doc_db(indexed_dataset_infos, "train")
    # <<<
