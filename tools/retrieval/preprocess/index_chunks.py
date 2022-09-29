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
import json
import numpy as np
import os
from pathlib import Path
import threading
import torch
from tqdm import tqdm

# from megatron import (
#     # get_args,
#     get_tokenizer,
# )
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.tokenizer.tokenizer import (
    _BertWordPieceTokenizer,
    _GPT2BPETokenizer,
)

from .utils import (
    get_individual_chunk_index_path,
    get_full_chunk_index_path,
    get_sampled_chunk_index_path,
)

# >>>
from lutil import pax
# <<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# see: notebook/faiss/create_chunks.ipynb
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def get_sorted_dataset_metadatas(args, workdir):

    assert len(args.data_path) % 2 == 0, \
        "currently, only blendable dataset is supported."

    # Data metadata.
    data_metas = []
    for i in range(0, len(args.data_path), 2):
        ratio = float(args.data_path[i])
        prefix = args.data_path[i + 1]
        path = prefix + ".bin"
        name = os.path.basename(prefix)
        assert os.path.exists(path)
        data_metas.append({
            "ratio" : ratio,
            "prefix" : prefix,
            "path" : path,
            "name" : name,
            "chunk_index_path" : get_individual_chunk_index_path(workdir, name)
        })

    # Deterministic dataset order (alphabetical).
    data_metas.sort(key = lambda m : m["prefix"])

    return data_metas


def save_dataset_metadatas(workdir, data_metas):
    """Save dataset order."""

    order_path = os.path.join(workdir, "order.json")
    with open(order_path, "w") as f:
        json.dump(data_metas, f, indent = 4) # remove 'indent', once debugged.


def build_partial_chunk_index(
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
        print(" > building partial chunk index, proc %d / %d, docs %d:%d / %d."%(
            proc_id,
            n_procs,
            doc_start_id,
            doc_end_id,
            n_docs,
        ))

    # Progress bars (snapshot of overall progress).
    doc_id_iter = range(doc_start_id, doc_end_id)
    # doc_id_iter = range(doc_start_id, min(n_docs, doc_start_id+128)) # for debug
    pbar = tqdm(doc_id_iter) \
        if proc_id in progress_proc_ids else \
           doc_id_iter

    chunk_index_valid = []
    chunk_index_invalid = []
    for doc_id in pbar:

        try:
            pbar.set_description("proc %d / %d." % (proc_id, n_procs))
        except:
            pass

        doc = indexed_dataset.get(doc_id)
        eod_id = doc[-1]
        doc = doc[:-1] # remove 'eod' token
        doc_len = len(doc)

        chunk_start_idxs = list(range(0, doc_len, args.retrieval_chunk_len))
        chunk_end_idxs = [min(doc_len, s + args.retrieval_chunk_len)
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

            _chunk_index = chunk_index_invalid \
                if len(bert_token_ids) == 0 else \
                   chunk_index_valid
            _chunk_index.append((
                doc_id,
                chunk_start_idx,
                chunk_end_idx,
                len(bert_token_ids),
            ))

    # if proc_id == 0:
    #     pax({
    #         "chunk_index / len" : len(chunk_index),
    #         "chunk_index" : chunk_index[:100],
    #     })

    return proc_id, chunk_index_valid, chunk_index_invalid


def build_individual_chunk_index(args, indexed_dataset):

    # >>>
    gpt_tokenizer = _GPT2BPETokenizer(
        vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json",
        merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt",
    )
    bert_tokenizer = _BertWordPieceTokenizer(
        vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/roberta_mmap/vocab.txt",
        lower_case = True,
    )
    # <<<

    n_procs = 128 # 8, 128

    # executor_ty = concurrent.futures.ThreadPoolExecutor
    executor_ty = concurrent.futures.ProcessPoolExecutor
    with executor_ty(max_workers = n_procs) as executor:
        futures = []
        for proc_id in range(n_procs): # not true process id
            futures.append(executor.submit(
                build_partial_chunk_index,
                args,
                proc_id,
                n_procs,
                indexed_dataset,
                gpt_tokenizer,
                bert_tokenizer,
            ))
        partial_chunk_indexes = []
        for future in concurrent.futures.as_completed(futures):
            partial_chunk_indexes.append(future.result())

    partial_chunk_indexes.sort(key = lambda item : item[0]) # sort by proc_id
    chunk_index_valid = [item
                         for partial_chunk_index in partial_chunk_indexes
                         for item in partial_chunk_index[1]]
    chunk_index_invalid = [item
                           for partial_chunk_index in partial_chunk_indexes
                           for item in partial_chunk_index[2]]

    # pax({
    #     "partial_chunk_indexes" :
    #     [ "%d, len %d" % (p, len(ii)) for p, ii in partial_chunk_indexes ],
    #     "chunk_index / len" : len(chunk_index),
    #     "chunk_index / start" : chunk_index[:8],
    #     "chunk_index / end" : chunk_index[-8:],
    #     "empty berts" : sum(item[3] == 0 for item in chunk_index),
    # })

    print(' > converting chunk index to numpy.')
    chunk_index_valid = np.array(chunk_index_valid)
    chunk_index_invalid = np.array(chunk_index_invalid)

    return chunk_index_valid, chunk_index_invalid


def build_individual_chunk_indexes(args, workdir, data_metas):

    print(" > build individual chunk indexes.")
    for data_index, data_meta in enumerate(data_metas):

        chunk_index_path = data_meta["chunk_index_path"]

        if os.path.exists(chunk_index_path):
            continue

        print(" > building individual chunk index, dataset %d / %d ... '%s'." %
              (data_index, len(data_metas), data_meta["name"]))

        indexed_dataset = make_indexed_dataset(data_meta["prefix"], "mmap", True)
        chunk_index_valid, chunk_index_invalid = \
            build_individual_chunk_index(args, indexed_dataset)

        print(" > saving chunk index.")

        f = h5py.File(chunk_index_path, "w")
        dset = f.create_dataset("chunks_valid", data = chunk_index_valid)
        dset = f.create_dataset("chunks_invalid", data = chunk_index_invalid)
        f.close()

        print(" > finished saving chunk index.")

    # Set n_chunks_{valid,invalid}, n_chunks_sampled (for unambiguity).
    print(" > compute n_chunks_all, n_chunks_valid, n_chunks_sampled.")
    for data_index, data_meta in enumerate(data_metas):

        f = h5py.File(data_meta["chunk_index_path"], "r")
        data_meta["n_chunks_valid"] = len(f["chunks_valid"])
        data_meta["n_chunks_invalid"] = len(f["chunks_invalid"])
        f.close()

        data_meta["n_chunks_sampled"] = \
            int(round(args.retrieval_nchunks_sampled * data_meta["ratio"]))

        assert data_meta["n_chunks_sampled"] < data_meta["n_chunks_valid"]
        
        # pax({"data_meta": data_meta})

    print(" > compute document offsets.")
    doc_offset = 0
    for data_index, data_meta in enumerate(data_metas):

        f = h5py.File(data_meta["chunk_index_path"], "r")
        data_meta["doc_offset"] = doc_offset
        doc_offset += f["chunks_valid"][-1, 0].item()
        f.close()

    # pax({"doc_offsets": [ m["doc_offset"] for m in data_metas ]})


def build_full_chunk_index(args, workdir, data_metas):

    print(" > build full chunk index.")

    full_index_path = get_full_chunk_index_path(workdir)
    n_chunks = {
        "valid" : sum(m["n_chunks_valid"] for m in data_metas),
        "invalid" : sum(m["n_chunks_invalid"] for m in data_metas),
    }

    # pax({
    #     "n_chunks_valid" : n_chunks_valid,
    #     "n_chunks_invalid" : n_chunks_invalid,
    # })

    # Delete existing chunk index if incorrect size.
    if os.path.exists(full_index_path):

        f = h5py.File(full_index_path, "r")

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
            raise Exception("remove full?")
            os.remove(full_index_path)

    # Build full chunk index.
    if not os.path.exists(full_index_path):

        f = h5py.File(full_index_path, "w")

        for validity in "valid", "invalid":

            chunk_index = f.create_dataset(f"chunks_{validity}",
                                           (n_chunks[validity], 4),
                                           dtype = "uint64") # "i8")
            dataset_offsets = f.create_dataset(f"dataset_offsets_{validity}",
                                               (len(data_metas) + 1,),
                                               dtype = "uint64")
            n_written = f.create_dataset(f"n_written_{validity}",
                                         (1,),
                                         dtype = "uint64")
            n_written[0] = 0

            start_index = 0
            for data_index, data_meta in enumerate(data_metas):

                print(" > concatenating (%s) chunks, dataset %d / %d ... '%s'." %
                      (validity, data_index, len(data_metas), data_meta["name"]))

                g = h5py.File(data_meta["chunk_index_path"], "r")
                data = g[f"chunks_{validity}"]
                chunk_index[start_index:start_index + len(data)] = data
                start_index += len(data)
                dataset_offsets[data_index + 1] = start_index
                n_written[0] = start_index
                g.close()

        f.close()


def build_sampled_chunk_index(args, workdir, data_metas):

    print(" > build sampled chunk index.")

    sampled_index_path = get_sampled_chunk_index_path(workdir)
    n_chunks = sum(m["n_chunks_sampled"] for m in data_metas)

    # Delete existing chunk index if incorrect size.
    if os.path.exists(sampled_index_path):

        f = h5py.File(sampled_index_path)
        n_alloc = len(f["chunks"])           # total allocated
        n_written = f["n_written"][0].item() # total written
        f.close()

        if n_chunks != n_alloc or n_chunks != n_written:
            os.remove(sampled_index_path)

    # Build sampled chunk index.
    if not os.path.exists(sampled_index_path):

        f = h5py.File(sampled_index_path, "w")
        chunk_index = f.create_dataset("chunks_valid", (n_chunks, 4), dtype = "i8")
        dataset_offsets = f.create_dataset(
            "dataset_offsets_valid", (len(data_metas) + 1,), dtype = "uint64")
        n_written = f.create_dataset("n_written_valid", (1,), dtype = "uint64")
        n_written[0] = 0

        start_index = 0
        for data_index, data_meta in enumerate(data_metas):

            print(" > concatenating chunks, dataset %d / %d ... '%s'." %
                  (data_index, len(data_metas), data_meta["name"]))

            g = h5py.File(data_meta["chunk_index_path"], "r")
            data = g["chunks_valid"][:data_meta["n_chunks_sampled"]]
            chunk_index[start_index:start_index + len(data)] = data
            start_index += len(data)
            dataset_offsets[data_index + 1] = start_index
            n_written[0] = start_index
            g.close()

        f.close()


# def dump_document_order():
# def save_document_order(args, workdir):
def build_chunk_indexes(args, workdir):

    # >>>
    assert torch.distributed.get_rank() == 0, "single process operation."
    # +++
    # if torch.distributed.get_rank() == 0:
    #     return
    # <<<

    # Dataset metadata. (sorted, official order)
    data_metas = get_sorted_dataset_metadatas(args, workdir)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # create_data_softlinks(data_files)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Build chunk indexes.
    build_individual_chunk_indexes(args, workdir, data_metas)
    build_full_chunk_index(args, workdir, data_metas)
    build_sampled_chunk_index(args, workdir, data_metas)

    # Save dataset metadata. (fully annotated at this point)
    save_dataset_metadatas(workdir, data_metas)

    # >>>
    # f = h5py.File(get_full_chunk_index_path(workdir), "r")
    # g = h5py.File(get_sampled_chunk_index_path(workdir), "r")
    # pax({
    #     "full / chunks" : str(f["chunks"].shape),
    #     "sampled / chunks" : str(g["chunks"].shape),
    #     "full / offsets" : np.copy(f["dataset_offsets"]).tolist(),
    #     "sampled / offsets" : np.copy(g["dataset_offsets"]).tolist(),
        
    # })
    # <<<

    # raise Exception("finished creating chunks.")

# eof
