This directory contains a collection of tools for building an similarity search index for retrieval-augmented language models.

Please see 'tools/retrieval/examples/get_cmd.sh' for example usage. At the moment, environment variable NPROCS can either be manually set, or copied from SLURM_TASKS_PER_NODE, depending on if using an interactive or a batch run (see top of get_cmd.sh).

Key files:

- main.py (entry point)
- examples/get_cmd.sh (sample arguments for main.py)
- examples/run_main.sh (calls get_cmd.sh, main.py)
- examples/run_cluster.sh (calls run_main.sh)

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
