This directory contains a collection of tools for building the retrieval database and pretraining neighbors for Retro. This preprocessing pipeline is broken into 3 main stages:

1. **Build retrieval chunk database** : Used for retrieving neighbors and continuation chunks, which are then passed through the retrieval encoder.
2. **Build index for similarity search** : Train and build a similarity search index for querying chunk neighbors.
3. **Query pretraining neighbors** : For matching pretraining samples to database chunks. Neighbors are generated separately for training, validation, and test datasets.

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

See `examples/get_cmd.sh` for example usage.

Key files:

- `main.py` : Entry point.
- `examples/get_cmd.sh` : Example arguments for `main.py`.
- `examples/run_main.sh` : Calls `get_cmd.sh`, `main.py`.

Use `--retro-tasks` to move through the preprocessing pipeline.

- Build chunk database: `--retro-tasks db-build`
- Build index: `--retro-tasks index-build`
- Query neighbors: `--retro-tasks pretraining-query-neighbors`

Sample code flow:

- `main.py` : Entry point (e.g., using `--retro-tasks X`).
- `db/build.py` : Build retrieval database.
- `index/train.py` : Train index on subset of database.
- `index/add.py` : Add database chunks to index.
- `pretraining/query.py` : Query pretraining samples for database neighbors (saved to disk and used during pretraining).

<!-- ################ stages ################ -->
# Stages

### Build retrieval chunk database

This *database* (stored as a 2-D array, NOT a relational database) consists of a list of chunks (traditionally length 64) extracted from the original GPT token dataset. This is simply a consecutive, non-overlapping chunking of the token dataset. Chunking only takes place within a document, and therefore the final chunk of each document has length: 1 <= chunk_length <= max_chunk_length.

We discard chunks that would convert to an empty Bert sequence (rare case, happens ~1/100,000 chunks in our case), since we use Bert embeddings for building our index. Thus, the total number of chunks in the database will be slightly less than a naive calculation.

### Build index for similarity search

To match pretraining chunks to database chunks, a similarity search index must be built to perform this querying. We use Faiss (https://github.com/facebookresearch/faiss) for training and building this index. Generally, the index is trained on a subset of all chunks in the database (specified via `--retro-nchunks-sampled`). After training, all chunks are added into the index, to be available during querying.

Indexes only accept 1-D floating point vectors for training and adding, so each chunk must first be embedded before passing to the index for either training or adding. We use Bert embeddings for this purpose, and the embeddings are generated automatically within the pipeline.

### Query pretraining neighbors

To ensure fast Retro pretraining, the database neighbors for pretraining samples are pre-computed and saved to disk, for efficient access within the Retro dataset. In this stage, the pretraining datasets (training, validation, and test) are iterated, each sample is broken into chunks, and the chunks are used for querying the index. Similar to when building the index, each chunk is embedded (via Bert) before querying the index.

The saved neighbors are labeled with unique dataset properties (i.e., seed, sequence length, number of samples, etc.) to ensure the neighbors generated during preprocessing match the neighbors requested during pretraining.

<!-- ################ code structure ################ -->
# Code structure

<!-- ################ arguments ################ -->
# Arguments

<!-- ################ pretraining ################ -->
# Pretraining

- New retro args in arguments.py (add_retro_args).
- Most important arg is `--retro-add-retriever`.