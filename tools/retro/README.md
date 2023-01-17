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

### `tools/retro/main.py`

This is the main entry point for Retro preprocessing. Call `main.py --help` to see arguments. Additionally, some Retro arguments are in Megatron's core arguments, so also see `add_retro_args()` section of `megatron/arguments.py` for additional arguments. Two of the most important arguments to customize are `--retro-workdir` and `--retro-tasks`.

`--retro-workdir` sets where the preprocessing pipeline saves its datasets and configuration files. This argument should remain consistent for a full pass through the pipeline.

`--retro-tasks` sets which stages of preprocessing to perform. As mentioned previously, the three high-level stages are: 1) build retrieval database, 2) build search index, and 3) query pretraining neighbors. `--retro-tasks` can be used to either run the full pipeline, or run each of these stages in isolation. The latter case is useful for tuning compute resources for each stage. For example, index training utilizes GPUs and requires relatively less time, while querying neighbors uses the CPU and is a relatively slow process. Example tasks include:

- `--retro-tasks build` : Run entire preprocessing pipeline.
- `--retro-tasks db-build` : Build retrieval database.
- `--retro-tasks index-build` : Train and build search index.
- `--retro-tasks pretraining-query-neighbors` : Query pretraining neighbors.

Multiple tasks can be specified by separating with commas (e.g., `--retro-tasks db-build,index-build`). Additionally, various 'miscellaneous' tasks are currently including, primarily for validating data for each stage; these task names can be seen in `main.py`.

### `tools/retro/examples`

Example scripts for setting arguments and launch Retro preprocessing. The key files here are:

- `get_cmd.sh` : Sets up arguments and command for preprocessing. **Important note**: this script assumes a few environment variables are already set before it is called. Please see the `Environment vars.` section at the top of this file. Generally, environment variables must be set to determine the location of Retro workdirs, input datasets, and GPT and Bert model information.
- `run_main.sh` : Calls `get_cmd.sh` to get arguments, and then calls `main.py` to launch preprocessing.

### `tools/retro/db`

Build the retrieval chunk database. The key files here are:

- `build.py` : Entry point for building the database. This code is responsible for iterating the input datasets (i.e., `--data-path`), parsing each dataset into consecutive chunks, checking for empty Bert (Wordpiece) conversions, and storing this information to disk. Two databases are created: 1) the retrieval database, and 2) a sampled database used for training the search index.
- `dataset.py` : Defines database class, for iterating or accessing chunks in the database. Each chunk contains its tokens, Bert conversion length, and dataset index.

### `tools/retro/index`

Build the search index. The key files here are:

- `build.py` : Entry point for building the search index. First, the index is trained on the sampled chunk database (see above) by calling `train.py`, and then all chunks for the full database are added to the index by calling `add.py`. Note that training requires first embedding (using Bert) all chunks (a parallel operation), and then loading these embeddings and training the index (a sequential operation), so it's best to change one's compute setup after all chunks have been embedded and saved to disk.
- `indexes/faiss_base.py` : Wrapper class for building a Faiss index, following the standard `train()` and `add()` operations.
- `indexes/faiss_par_add.py` : Similar to above, except it uses an embarrassingly parallel (multi-node, multi-process) `add()` operation. Vectors are first added to separate index copies, and then merged together.

### `tools/retro/pretraining`

Query the pretraining datasets (training, validation, test) for their neighbors within the database. Neighbors are queried during preprocessing -- rather than during pretraining -- because querying is a fairly slow operation, so it would be a bottleneck if performed during pretraining. Queried neighbors are tagged with their unique identifying information (e.g., `train_indexmap_27662746ns_2048sl_1234s`), so as to avoid incorrect references during pretraining. The key files here are:

- `query.py` : Entry point for querying. The pretraining datasets are iterated, and each chunk within each sample is queried using the search index. These neighbors are filtered by discarding any database chunks that fall within the same document as any chunk within a pretraining sample.
- `chunk_dataset.py` : This creates an iterable 'chunk' dataset form of a pretraining dataset. This is just a light wrapper, but makes it easier to deterministically iterate and assign IDs to each chunk in a sample dataset.
- `retro_dataset.py` : The Retro dataset used for pretraining (not used in preprocessing). Each sample returns the sample tokens, along with neighbor tokens for each chunk within the sample.

### `tools/retro/cli`
### `tools/retro/utils`
### `tools/bert_embedding`

### Pretraining

- `pretrain_retro.py` : Launch script for pretraining Retro. Similar to `pretrain_gpt.py`, except this script handles loading neighbor tokens and setting up the neighbor attention mask.
<!-- - `megatron/data/gpt_dataset.py` : ? -->
- `megatron/model/retro_transformer.py` : Implementation of Retro model, including the main transformer, the retrieval encoder, and chunked cross-attention layers. Note that currently, `retro_transformer.py` contains several classes that are nearly identical to `transformer.py`, except for 1 or 2 lines, due to code changes that are yet to be integrated.


<!-- ################ arguments ################ -->
# Arguments

<!-- ################ pretraining ################ -->
# Pretraining

- New retro args in arguments.py (add_retro_args).
- Most important arg is `--retro-add-retriever`.