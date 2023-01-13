This directory contains a collection of tools for building the retrieval database and pretraining neighbors for Retro. This preprocessing pipeline is broken into 3 main stages:

1. **Build retrieval chunk database** : Used for retrieving neighbors and continuation chunks, which are then passed through the retrieval encoder.
2. **Build index for similarity search** : Train and build a similarity search index for querying chunk neighbors.
3. **Query pretraining neighbors** : For matching pretraining samples to database chunks. Neighbors are generated separately for training and validation datasets.

The following overview goes into more detail on the pipeline, code structure, usage, and pretraining.

<!-- ################ contents ################ -->
# Contents

  * [Quick start](#quick-start)
  * [Stages](#stages)
  * [Code structure](#code-structure)
  * [Arguments](#arguments)
  * [Pretraining](#pretraining)

<!-- ################ quick start ################ -->
# Quick start

<!-- ################ stages ################ -->
# Stages

## Build retrieval chunk database

This *database* (stored as a 2-D array, NOT a relational database) consists of a list of chunks (traditionally length 64) extracted from the original GPT token dataset. This is simply a consecutive, non-overlapping chunking of the token dataset. Chunking only takes place within a document, and therefore the final chunk of each document has length: 1 <= chunk_length <= max_chunk_length.

We discard chunks that would convert to an empty Bert sequence (rare case, happens ~1/100,000 chunks in our case), since we use Bert embeddings for building our index. Thus, the total number of chunks in the database will be slightly less than a naive calculation.

## Build index for similarity search

To match pretraining chunks to database chunks, a similarity search index must be built to perform this querying. We use Faiss (https://github.com/facebookresearch/faiss) for training and building this index. Generally, the index is trained on a subset of all chunks in the database (specified via `--retro-nchunks-sampled`). After training, all chunks are added into the index, to be available during querying.

Indexes only accept 1-D floating point vectors for training and adding, so each chunk must first be embedded before passing to the index for either training or adding. We use Bert embeddings for this purpose, and the embeddings are generated automatically within the pipeline.

## Query pretraining neighbors

To ensure fast Retro pretraining

After preprocessing has finished, the user-specified output directory will contain:

- **Retrieval chunk database** : 
- **Pretraining neighbor indexes** : 


<!-- ################ code structure ################ -->
# Code structure

Please see 'tools/retrieval/examples/get_cmd.sh' for example usage. At the moment, environment variable NPROCS can either be manually set, or copied from SLURM_TASKS_PER_NODE, depending on if using an interactive or a batch run (see top of get_cmd.sh).

Key files:

- main.py (entry point)
- examples/get_cmd.sh (example arguments for main.py)
- examples/run_main.sh (calls get_cmd.sh, main.py)

Currently working indexes:

- FaissBaseIndex (--index-ty faiss-base)
- FaissParallelAddIndex (--index-ty faiss-par-add)
- [not recently tested] FaissDecompIndex (--index-ty faiss-decomp)

Example tasks (use with --tasks):

- [no] Bert embeddings ('embed')
- [yes] Train ('train')
- [yes] Add ('add', 'remove-add-outputs')
- [needs cleanup] Query ('query', 'plot-acc', 'query-flat-nns')
- [yes] Verify ('verify-codes', 'verify-nbrs')
  - note: 'verify-nbrs' currently requires 'query_index.py' to be manually run first

Sample code flow:
- main.py (entry point; e.g., using '--tasks add')
- add/add.py ('add' pipeline; init index; call index.add())
- index/faiss_par_add/__init__.py (run add in parallel; store intermediate/final index to args.index_dir_path)

<!-- ################ arguments ################ -->
<!-- ################ pretraining ################ -->
