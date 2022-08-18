This directory contains a collection of tools for building an similarity search index for retrieval-augmented language models.

Please see 'tools/retrieval/examples/get_cmd.sh' for example usage. At the moment, environment variable NPROCS can either be manually set, or copied from SLURM_TASKS_PER_NODE, depending on if using an interactive or a batch run (see top of get_cmd.sh).

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
