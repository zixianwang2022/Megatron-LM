# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
from functools import reduce
import h5py
import numpy as np
import os
from pathlib import Path
import threading
import torch
from tqdm import tqdm

from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)
from tools.retro.utils import get_gpt_tokenizer, get_bert_tokenizer

from .utils import (
    get_individual_db_dir,
    get_individual_db_path,
    # get_full_db_info,
    # get_sampled_db_info,
    get_db_info_map,
    save_indexed_dataset_infos,
)

# >>>
from lutil import pax
# <<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# see: notebook/faiss/create_chunks.ipynb
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# def get_sorted_dataset_metadatas(args, workdir):
def get_sorted_indexed_dataset_infos(args):

    assert len(args.data_path) % 2 == 0, \
        "currently, only blendable dataset is supported."

    # Data metadata.
    infos = []
    for i in range(0, len(args.data_path), 2):
        ratio = float(args.data_path[i])
        prefix = args.data_path[i + 1]
        path = prefix + ".bin"
        name = os.path.basename(prefix)
        assert os.path.exists(path)
        infos.append({
            "ratio" : ratio,
            "prefix" : prefix,
            "path" : path,
            "name" : name,
            "db_path" : get_individual_db_path(args, name),
        })

    # Deterministic dataset order (alphabetical).
    infos.sort(key = lambda m : m["prefix"])

    # pax(0, {"infos": infos, "infos / 0": infos[0]})

    return infos


# def build_partial_chunk_db(
def build_partial_db(
        args,
        proc_id,
        n_procs,
        indexed_dataset,
        gpt_tokenizer,
        bert_tokenizer,
):

    # progress_proc_ids = set(range(0, n_procs, int(n_procs / 8)))
    progress_proc_ids = set(range(n_procs))

    # n_docs = len(indexed_dataset)
    n_docs = len(indexed_dataset.doc_idx) - 1 # doc_idx starts at 0
    n_docs_per_proc = int(np.ceil(n_docs / n_procs))
    doc_start_id = proc_id * n_docs_per_proc
    doc_end_id = min(n_docs, doc_start_id + n_docs_per_proc)

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
    # doc_id_iter = range(doc_start_id, min(n_docs, doc_start_id+10)) # for debug
    pbar = tqdm(doc_id_iter) \
        if proc_id in progress_proc_ids else \
           doc_id_iter

    chunk_db_valid = []
    chunk_db_invalid = []
    for doc_id in pbar:

        try:
            pbar.set_description("proc %d / %d." % (proc_id, n_procs))
        except:
            pass

        doc = indexed_dataset.get(doc_id)
        eod_id = doc[-1]
        doc = doc[:-1] # remove 'eod' token
        doc_len = len(doc)

        chunk_start_idxs = list(range(0, doc_len, args.retro_gpt_chunk_length))
        chunk_end_idxs = [min(doc_len, s + args.retro_gpt_chunk_length)
                          for s in chunk_start_idxs]

        for i, chunk_start_idx in enumerate(chunk_start_idxs):
            chunk_end_idx = chunk_end_idxs[i]
            gpt_token_ids = indexed_dataset.get(
                idx = doc_id,
                offset = chunk_start_idx,
                length = chunk_end_idx - chunk_start_idx,
            )
            gpt_token_ids = [ t for t in gpt_token_ids.tolist() if t != eod_id ]
            text = gpt_tokenizer.detokenize(gpt_token_ids)
            bert_token_ids = bert_tokenizer.tokenize(text)

            _chunk_db = chunk_db_invalid \
                if len(bert_token_ids) == 0 else \
                   chunk_db_valid
            _chunk_db.append((
                doc_id,
                chunk_start_idx,
                chunk_end_idx,
                len(bert_token_ids),
            ))

    # if proc_id == 0:
    #     pax({
    #         "chunk_db / len" : len(chunk_db),
    #         "chunk_db" : chunk_db[:100],
    #     })

    return proc_id, chunk_db_valid, chunk_db_invalid


# def build_individual_chunk_db(args, indexed_dataset):
def build_individual_db(args, gpt_tokenizer, bert_tokenizer, indexed_dataset):

    n_procs = 128 # 8, 128

    # executor_ty = concurrent.futures.ThreadPoolExecutor # slow.
    executor_ty = concurrent.futures.ProcessPoolExecutor
    with executor_ty(max_workers = n_procs) as executor:
        futures = []
        for proc_id in range(n_procs): # not true process id
            futures.append(executor.submit(
                build_partial_db,
                args,
                proc_id,
                n_procs,
                indexed_dataset,
                gpt_tokenizer,
                bert_tokenizer,
            ))
        partial_chunk_dbs = []
        for future in concurrent.futures.as_completed(futures):
            partial_chunk_dbs.append(future.result())

    partial_chunk_dbs.sort(key = lambda item : item[0]) # sort by proc_id
    chunk_db_valid = [item
                      for partial_chunk_db in partial_chunk_dbs
                      for item in partial_chunk_db[1]]
    chunk_db_invalid = [item
                        for partial_chunk_db in partial_chunk_dbs
                        for item in partial_chunk_db[2]]

    # pax({
    #     "partial_chunk_dbs" :
    #     [ "%d, len %d" % (p, len(ii)) for p, ii in partial_chunk_dbs ],
    #     "chunk_db / len" : len(chunk_db),
    #     "chunk_db / start" : chunk_db[:8],
    #     "chunk_db / end" : chunk_db[-8:],
    #     "empty berts" : sum(item[3] == 0 for item in chunk_db),
    # })

    print(' > converting chunk db to numpy.')
    chunk_db_valid = np.array(chunk_db_valid)
    chunk_db_invalid = np.array(chunk_db_invalid)

    return chunk_db_valid, chunk_db_invalid


# def build_individual_chunk_dbs(args, workdir, data_metas):
def build_individual_dbs(args, indexed_dataset_infos):

    # Individual workdir.
    individual_dir = get_individual_db_dir(args)
    os.makedirs(individual_dir, exist_ok = True)

    # Tokenizers.
    gpt_tokenizer = get_gpt_tokenizer()
    bert_tokenizer = get_bert_tokenizer()

    # Build individual dbs.
    print(" > build individual chunk dbs.")
    for ds_index, ds_info in enumerate(indexed_dataset_infos):

        db_path = ds_info["db_path"]

        if os.path.exists(db_path):
            continue

        print(" > building individual db, dataset %d / %d ... '%s'." % (
            ds_index,
            len(indexed_dataset_infos),
            ds_info["name"],
        ))

        indexed_dataset = make_indexed_dataset(ds_info["prefix"], "mmap", True)
        db_valid, db_invalid = build_individual_db(args,
                                                   gpt_tokenizer,
                                                   bert_tokenizer,
                                                   indexed_dataset)

        print(" > saving individual db.")
        f = h5py.File(db_path, "w")
        dset = f.create_dataset("chunks_valid", data = db_valid)
        dset = f.create_dataset("chunks_invalid", data = db_invalid)
        f.close()

        print(" > finished saving individual db.")

    # Set n_chunks_{valid,invalid}, n_chunks_sampled (for unambiguity).
    print(" > compute n_chunks_all, n_chunks_valid, n_chunks_sampled.")
    for ds_index, ds_info in enumerate(indexed_dataset_infos):

        f = h5py.File(ds_info["db_path"], "r")
        ds_info["n_chunks_valid"] = len(f["chunks_valid"])
        ds_info["n_chunks_invalid"] = len(f["chunks_invalid"])
        f.close()

        ds_info["n_chunks_sampled"] = \
            int(round(args.retro_nchunks_sampled * ds_info["ratio"]))

        assert ds_info["n_chunks_sampled"] < ds_info["n_chunks_valid"]
        
        # pax({"ds_info": ds_info})

    # Compute document offsets.
    print(" > compute document offsets.")
    doc_offset = 0
    for ds_index, ds_info in enumerate(indexed_dataset_infos):

        f = h5py.File(ds_info["db_path"], "r")
        ds_info["doc_offset"] = doc_offset
        doc_offset += f["chunks_valid"][-1, 0].item()
        f.close()

    # pax({"doc_offsets": [ i["doc_offset"] for i in indexed_dataset_infos ]})


# def build_full_chunk_db(args, workdir, data_metas):
def build_full_db(args, indexed_dataset_infos):

    print(" > build full chunk db.")

    # pax({
    #     "ds_infos" : indexed_dataset_infos,
    #     "ds_infos / 0" : indexed_dataset_infos[0],
    #     "ds_infos / 1" : indexed_dataset_infos[1],
    # })

    # full_db_path = get_full_db_info(args)["db_path"]
    full_db_path = get_db_info_map(args)["full"]["db_path"]
    n_chunks = {
        "valid" : sum(m["n_chunks_valid"] for m in indexed_dataset_infos),
        "invalid" : sum(m["n_chunks_invalid"] for m in indexed_dataset_infos),
    }

    # pax({"n_chunks": n_chunks})

    # Delete existing chunk db if incorrect size.
    if os.path.exists(full_db_path):

        try:

            f = h5py.File(full_db_path, "r")

            # Total allocated.
            n_alloc_valid = len(f["chunks_valid"])
            n_alloc_invalid = len(f["chunks_invalid"])

            # Total written.
            n_written_valid = f["n_written_valid"][0].item()
            n_written_invalid = f["n_written_invalid"][0].item()

            f.close()

            if n_chunks["valid"] != n_alloc_valid or \
               n_chunks["valid"] != n_written_valid or \
               n_chunks["invalid"] != n_alloc_invalid or \
               n_chunks["invalid"] != n_written_invalid:
                os.remove(full_db_path)

        except Exception as e:
            if isinstance(e, OSError):
                os.remove(full_db_path)
            elif isinstance(e, KeyError):
                f.close()
                os.remove(full_db_path)
            else:
                raise e

    if os.path.exists(full_db_path):

        f = h5py.File(full_db_path, "r")

        # Total allocated.
        n_alloc_valid = len(f["chunks_valid"])
        n_alloc_invalid = len(f["chunks_invalid"])

        # Total written.
        n_written_valid = f["n_written_valid"][0].item()
        n_written_invalid = f["n_written_invalid"][0].item()
            
        f.close()

        if n_chunks["valid"] != n_alloc_valid or \
           n_chunks["valid"] != n_written_valid or \
           n_chunks["invalid"] != n_alloc_invalid or \
           n_chunks["invalid"] != n_written_invalid:
            os.remove(full_db_path)

    # Build full chunk db.
    if not os.path.exists(full_db_path):

        os.makedirs(os.path.dirname(full_db_path), exist_ok = True)
        f = h5py.File(full_db_path, "w")

        for validity in "valid", "invalid":

            chunk_db = f.create_dataset(f"chunks_{validity}",
                                           (n_chunks[validity], 4),
                                           dtype = "uint64") # "i8")
            dataset_offsets = f.create_dataset(f"dataset_offsets_{validity}",
                                               (len(indexed_dataset_infos) + 1,),
                                               dtype = "uint64")
            n_written = f.create_dataset(f"n_written_{validity}",
                                         (1,),
                                         dtype = "uint64")
            n_written[0] = 0

            start_index = 0
            for ds_index, ds_info in enumerate(indexed_dataset_infos):

                print(" > concatenating (%s) chunks, dataset %d / %d ... '%s'." %
                      (validity, ds_index,
                       len(indexed_dataset_infos), ds_info["name"]))

                g = h5py.File(ds_info["db_path"], "r")
                data = g[f"chunks_{validity}"]
                chunk_db[start_index:start_index + len(data)] = data
                start_index += len(data)
                dataset_offsets[ds_index + 1] = start_index
                n_written[0] = start_index
                g.close()

        f.close()


# def build_sampled_chunk_db(args, workdir, data_metas):
def build_sampled_db(args, indexed_dataset_infos):

    print(" > build sampled chunk db.")

    # sampled_db_path = get_sampled_db_info(args)["db_path"]
    sampled_db_path = get_db_info_map(args)["sampled"]["db_path"]
    n_chunks = sum(m["n_chunks_sampled"] for m in indexed_dataset_infos)

    # Delete existing chunk db if incorrect size.
    if os.path.exists(sampled_db_path):

        try:

            f = h5py.File(sampled_db_path)
            n_alloc = len(f["chunks_valid"])           # total allocated
            n_written = f["n_written_valid"][0].item() # total written
            f.close()

            if n_chunks != n_alloc or n_chunks != n_written:
                os.remove(sampled_db_path)

        except Exception as e:
            if isinstance(e, OSError):
                os.remove(full_db_path)
            elif isinstance(e, KeyError):
                f.close()
                os.remove(full_db_path)
            else:
                raise e

    # Build sampled chunk db.
    if not os.path.exists(sampled_db_path):

        os.makedirs(os.path.dirname(sampled_db_path), exist_ok = True)
        f = h5py.File(sampled_db_path, "w")
        chunk_db = f.create_dataset("chunks_valid", (n_chunks, 4), dtype = "i8")
        dataset_offsets = f.create_dataset(
            "dataset_offsets_valid", (len(indexed_dataset_infos) + 1,), dtype = "uint64")
        n_written = f.create_dataset("n_written_valid", (1,), dtype = "uint64")
        n_written[0] = 0

        start_index = 0
        for ds_index, ds_info in enumerate(indexed_dataset_infos):

            print(" > concatenating chunks, dataset %d / %d ... '%s'." %
                  (ds_index, len(indexed_dataset_infos), ds_info["name"]))

            g = h5py.File(ds_info["db_path"], "r")
            data = g["chunks_valid"][:ds_info["n_chunks_sampled"]]
            chunk_db[start_index:start_index + len(data)] = data
            start_index += len(data)
            dataset_offsets[ds_index + 1] = start_index
            n_written[0] = start_index
            g.close()

        f.close()


# def dump_document_order():
# def save_document_order(args, workdir):
# def build_chunk_dbs(args, workdir):
# def preprocess_chunk_db(args, workdir):
def preprocess_db(args, timer):

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # create_data_softlinks(data_files)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Single process here, since we use ProcessPoolExecutor.
    if torch.distributed.get_rank() != 0: # Process
        return

    # Dataset metadata. (sorted, official order)
    # individual_workdir = os.path.join(workdir, "chunk_dbs")
    # individual_workdir = get_individual_db_dir(args)
    # os.makedirs(individual_workdir, exist_ok = True)

    indexed_dataset_infos = get_sorted_indexed_dataset_infos(args)

    # Build dbs.
    build_individual_dbs(args, indexed_dataset_infos)
    build_full_db(args, indexed_dataset_infos)
    build_sampled_db(args, indexed_dataset_infos)

    # Save (fully annotated) indexed dataset infos.
    save_indexed_dataset_infos(args, indexed_dataset_infos)

