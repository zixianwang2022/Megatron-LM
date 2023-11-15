# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Preprocess data for Retro.

Stages (see argument '--retro-tasks'):
- Build chunk database (DB).
- Build index (train, add).
- Query pretraining neighbors.
"""

from dataclasses import dataclass
# import json
# import os
# import torch

# from megatron import get_args, initialize_megatron, print_rank_0
# from megatron.global_vars import set_retro_args
# from megatron.core.models.retro.data.db import build_db
# from megatron.core.models.retro.data.index import add_to_index, build_index, train_index
# from megatron.core.models.retro.data.query import query_pretraining_neighbors
# from megatron.core.models.retro.data.utils import get_args_path
from megatron.core.transformer import TransformerConfig

# >>>
from lutil import pax
# <<<


        # retro_workdir (str): Retro working directory, which contains the
        #     preprocessed data for for pretraining. This directory is built during
        #     preprocessing (see tools/retro/README.md), and contains subdirectories
        #     for the chunk database and pretraining neighbors.
@dataclass
class RetroPreprocessingConfig(TransformerConfig):

    """Configuration object for Retro preprocessing.

    *Note* : Arguments prefixed with '--retro-gpt-*' or '--retro-bert-*' are
    included and named as such to more easily handle managing both models
    running at the same time. Megatron is not optimized to run two models at
    once, so this naming convention makes it clearer.

    Attributes:

        retro_tasks (str): Comma-separated list of tasks to run. Run entire
            preprocesing pipeline by using '--retro-tasks build'. Alternatively,
            run individual stages with tasks (in this order) 'db-build',
            'index-build', or 'query-pretraining-neighbors'. For example,
            '--retro-tasks db-build,index-build,query-pretraining-neighbors' is
            equivalent to '--retro-tasks build'; or the argument can contain a
            subset of these tasks. Stages must always be run in the correct order
            (listed above).
        retro_block_size (int): Number of chunks to process at a time when
            generating Bert embeddings and querying the search index. Partial
            results for each block are generally saved to disk in separate files.
        retro_doc_block_size (int): Number of documents to processe at time when
            processing token datasets into chunk databases. The partial chunk
            database for each block is saved into a separate file.
        retro_gpt_seed (int): Random seed used for python, numpy, pytorch, and
            cuda.
        retro_gpt_data_path (str): Path to the training dataset. Accepted format:
            1) a single data path, 2) multiple datasets in the form:
            dataset1-weight dataset1-path dataset2-weight dataset2-path ... It is
            used with --split when a single dataset used for all three: train,
            valid and test. It is exclusive to the other --*-data-path args.
        retro_gpt_data_cache_path (str): Path to a directory to hold cached index
            files.
        retro_gpt_split (str): Comma-separated list of proportions for training,
            validation, and test split. For example the split `90,5,5` will use
            90%% of data for training, 5%% for validation and 5%% for test.
        retro_gpt_eval_interval (int): GPT evaluation interval.
        retro_gpt_eval_iters (int): GPT evaluation iterations.
        retro_gpt_tokenizer_type (str): GPT tokenizer type.
        retro_gpt_vocab_file (str): GPT vocab file.
        retro_gpt_merge_file (str): GPT merge file.
        retro_gpt_tokenizer_model (str): GPT tokenizer model file.
        retro_gpt_seq_length (int): GPT sequence length.
        retro_gpt_global_batch_size (int): GPT global batch size.
        retro_gpt_chunk_length (int): GPT chunk length.
        retro_bert_vocab_file (str): Bert vocab file.
        retro_bert_tokenizer_type (str): Bert tokenizer type (for when using
            '--bert-embedder-type megatron').
        retro_bert_batch_size (int): Micro-batch size for processing Bert
            embeddings.
        retro_bert_max_chunk_length (int): Maximum sequence length for Bert
            embeddings. (Named 'chunk' here in reference to these Bert sequences
            being converted from GPT chunks.)
        retro_index_nfeats (int): Dimension of Bert embeddings. Bert-large is
            commonly used, so this value defaults to 1024.
        retro_index_type (str): A 'faiss-base' index is a simple, un-optimized
            wrapper around a Faiss index. A 'faiss-par-add' index optimizes the
            'add()' method by making it multi-node and multi-process, but with
            bit-wise equivalent results.
        retro_index_str (str): Index string used for calling
            faiss.index_factory(). For example, 'IVF262144_HNSW32,Flat' or
            'OPQ32_256,IVF4194304_HNSW32,PQ32'.
        retro_index_ntrain (int): Number of database chunks to use for training
            the index. This value must be less or equal to the total number of
            chunks in the database.
        retro_index_train_load_fraction (float): Fraction of sampled chunks to
            use for training the index. Useful when our total sampled embeddings
            use too much memory; lowering the load fraction is less costly than
            re-embedding a new sampled dataset from scratch.
        retro_index_add_load_fraction (float): Fraction of database chunks to use
            for adding to the index. Useful when our total index size would use
            too much memory; lowering the load fraction is less costly than
            re-designing our token datasets.
        retro_index_delete_training_embeddings (bool): Delete training embeddings
            for the search index. Useful for debugging.
        retro_index_delete_added_codes (bool): Delete added codes for the search
            index. Useful for debugging.
        retro_query_ef_search (int): Index ef-search parameter for HNSW during
            querying.
        retro_query_nprobe (int): Index nprobe parameter for IVF during querying.
        retro_query_num_neighbors_query (int): Number of neighbors to retrieve
            when calling index.search().
        retro_query_num_neighbors_save (int): Number of neighbors to save to disk
            after the index's returned neighbors. If longer than target value,
            neighbors truncated; and if shorter than target value, neighbors are
            padded with -1's.
    """

    # Basic.
    # >>>
    # retro_workdir: str = None
    # <<<
    retro_tasks: str = 'build'
    retro_block_size: int = 100000
    retro_doc_block_size: int = 100000

    # GPT.
    retro_gpt_seed: int = 1234
    retro_gpt_data_path: list = None
    retro_gpt_data_cache_path: str = None
    retro_gpt_split: str = '969,30,1'
    retro_gpt_eval_interval: int = None
    retro_gpt_eval_iters: int = None
    retro_gpt_tokenizer_type: str = None
    retro_gpt_vocab_file: str = None
    retro_gpt_merge_file: str = None
    retro_gpt_tokenizer_model: str = None
    retro_gpt_seq_length: int = None
    retro_gpt_global_batch_size: int = None
    retro_gpt_chunk_length: int = 64

    # Bert.
    retro_bert_vocab_file: str = None
    retro_bert_tokenizer_type: str = None
    retro_bert_batch_size: int = 128
    retro_bert_max_chunk_length: int = 256

    # Index.
    retro_index_nfeats: int = 1024
    retro_index_type: str = 'faiss-par-add'
    retro_index_str: str = None
    retro_index_ntrain: int = None
    retro_index_train_load_fraction: float = 1.0
    retro_index_add_load_fraction: float = 1.0
    # >>>
    # retro_index_no_delete_training_embeddings: bool = True
    # retro_index_no_delete_added_codes: bool = True
    retro_index_delete_training_embeddings: bool = True
    retro_index_delete_added_codes: bool = True
    # <<<

    # Query.
    retro_query_ef_search: int = 256
    retro_query_nprobe: int = 65536
    retro_query_num_neighbors_query: int = 200
    retro_query_num_neighbors_save: int = 20

    def __post_init__(self):

        # >>>
        # # Enforce argument naming convention.
        # for action in group._group_actions:
        #     prefix = action.dest.split("_")[0]
        #     assert prefix == "retro", \
        #         "Retro args must be prefixed with '--retro-*', for consistent " \
        #         "styling. Please fix '%s'." % ", ".join(action.option_strings)
        # <<<

        # Split retro tasks.
        self.retro_tasks = self.retro_tasks.split(",")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def add_retro_args(parser):
#     """Retro preprocesing arguments.

#     *Note* : Arguments prefixed with '--retro-gpt-*' or '--retro-bert-*' are
#     included and named as such to more easily handle managing both models
#     running at the same time. Megatron is not optimized to run two models at
#     once, so this naming convention makes it clearer.
#     """

#     group = parser.add_argument_group(title="Retro preprocessing.")

#     # Basic args.
#     group.add_argument("--retro-tasks", default="build",
#                        help="Comma-separated list of tasks to run. Run entire "
#                        "preprocesing pipeline by using '--retro-tasks build'. "
#                        "Alternatively, run individual stages with tasks (in "
#                        "this order) 'db-build', 'index-build', or "
#                        "'query-pretraining-neighbors'. For example, "
#                        "'--retro-tasks db-build,index-build,"
#                        "query-pretraining-neighbors' is equivalent to "
#                        "'--retro-tasks build'; or the argument can contain "
#                        "a subset of these tasks. Stages must always be run "
#                        "in the correct order (listed above).")
#     group.add_argument("--retro-block-size", type=int, default=100000,
#                        help="Number of chunks to process at a time when "
#                        "generating Bert embeddings and querying the search "
#                        "index. Partial results for each block are generally "
#                        "saved to disk in separate files.")
#     group.add_argument("--retro-doc-block-size", type=int, default=100000,
#                        help="Number of documents to processe at time when "
#                        "processing token datasets into chunk databases. The "
#                        "partial chunk database for each block is saved into "
#                        "a separate file.")

#     # GPT args.
#     group.add_argument('--retro-gpt-seed', type=int, default=1234,
#                        help='Random seed used for python, numpy, '
#                        'pytorch, and cuda.')
#     group.add_argument('--retro-gpt-data-path', nargs='*', required=True,
#                        help='Path to the training dataset. Accepted format:'
#                        '1) a single data path, 2) multiple datasets in the'
#                        'form: dataset1-weight dataset1-path dataset2-weight '
#                        'dataset2-path ... It is used with --split when a '
#                        'single dataset used for all three: train, valid '
#                        'and test. It is exclusive to the other '
#                        '--*-data-path args')
#     group.add_argument('--retro-gpt-split', type=str, default='969,30,1',
#                        help='Comma-separated list of proportions for training,'
#                        ' validation, and test split. For example the split '
#                        '`90,5,5` will use 90%% of data for training, 5%% for '
#                        'validation and 5%% for test.')
#     group.add_argument("--retro-gpt-eval-interval", type=int, required=True,
#                        help="GPT evaluation interval.")
#     group.add_argument("--retro-gpt-eval-iters", type=int, required=True,
#                        help="GPT evaluation iterations.")
#     group.add_argument("--retro-gpt-tokenizer-type", required=True,
#                        help="GPT tokenizer type.")
#     group.add_argument("--retro-gpt-vocab-file", help="GPT vocab file.")
#     group.add_argument("--retro-gpt-merge-file", help="GPT merge file.")
#     group.add_argument("--retro-gpt-tokenizer-model",
#                        help="GPT tokenizer model file.")
#     group.add_argument("--retro-gpt-seq-length", type=int, required=True,
#                        help="GPT sequence length.")
#     group.add_argument("--retro-gpt-global-batch-size", type=int, required=True,
#                        help="GPT global batch size.")
#     group.add_argument("--retro-gpt-chunk-length", type=int, default=64,
#                        help="GPT chunk length.")

#     # Bert args.
#     group.add_argument("--retro-bert-vocab-file", required=True,
#                        help="Bert vocab file.")
#     group.add_argument("--retro-bert-tokenizer-type", required=True,
#                        help="Bert tokenizer type (for when using "
#                        "'--bert-embedder-type megatron').")
#     group.add_argument("--retro-bert-batch-size", type=int, default=128,
#                        help="Micro-batch size for processing Bert embeddings.")
#     group.add_argument("--retro-bert-max-chunk-length", type=int, default=256,
#                        help="Maximum sequence length for Bert embeddings. "
#                        "(Named 'chunk' here in reference to these Bert "
#                        "sequences being converted from GPT chunks.)")

#     # Index args.
#     group.add_argument("--retro-index-nfeats", "-f", type=int, default=1024,
#                        help="Dimension of Bert embeddings. Bert-large is "
#                        "commonly used, so this value defaults to 1024.")
#     group.add_argument("--retro-index-type", default="faiss-par-add",
#                        choices=["faiss-base", "faiss-par-add"],
#                        help="A 'faiss-base' index is a simple, un-optimized "
#                        "wrapper around a Faiss index. A 'faiss-par-add' index "
#                        "optimizes the 'add()' method by making it multi-node "
#                        "and multi-process, but with bit-wise equivalent "
#                        "results.")
#     group.add_argument("--retro-index-str", required=True,
#                        help="Index string used for calling "
#                        "faiss.index_factory(). For example, "
#                        "'IVF262144_HNSW32,Flat' or "
#                        "'OPQ32_256,IVF4194304_HNSW32,PQ32'.")
#     group.add_argument("--retro-index-ntrain", type=int, required=True,
#                        help="Number of database chunks to use for training "
#                        "the index. This value must be less or equal to the "
#                        "total number of chunks in the database.")
#     group.add_argument("--retro-index-train-load-fraction",
#                        type=float, default=1.,
#                        help="Fraction of sampled chunks to use for training "
#                        "the index. Useful when our total sampled embeddings "
#                        "use too much memory; lowering the load fraction is "
#                        "less costly than re-embedding a new sampled dataset "
#                        "from scratch.")
#     group.add_argument("--retro-index-add-load-fraction",
#                        type=float, default=1.,
#                        help="Fraction of database chunks to use for adding to "
#                        "the index. Useful when our total index size would "
#                        "use too much memory; lowering the load fraction is "
#                        "less costly than re-designing our token datasets.")
#     group.add_argument("--retro-index-no-delete-training-embeddings",
#                        action='store_false',
#                        dest="retro_index_delete_training_embeddings",
#                        help="Skip deleting training embeddings for the search "
#                        "index. Useful for debugging.")
#     group.add_argument("--retro-index-no-delete-added-codes",
#                        action='store_false',
#                        dest="retro_index_delete_added_codes",
#                        help="Skip deleting added codes for the search "
#                        "index. Useful for debugging.")

#     # Query args.
#     group.add_argument("--retro-query-ef-search", type=int, default=256,
#                        help="Index ef-search parameter for HNSW during querying.")
#     group.add_argument("--retro-query-nprobe", type=int, default=65536,
#                        help="Index nprobe parameter for IVF during querying.")
#     group.add_argument("--retro-query-num-neighbors-query", type=int, default=200,
#                        help="Number of neighbors to retrieve when calling "
#                        "index.search().")
#     group.add_argument("--retro-query-num-neighbors-save", type=int, default=20,
#                        help="Number of neighbors to save to disk after "
#                        "the index's returned neighbors. If longer than target "
#                        "value, neighbors truncated; and if shorter than target "
#                        "value, neighbors are padded with -1's.")

#     # Enforce argument naming convention.
#     for action in group._group_actions:
#         prefix = action.dest.split("_")[0]
#         assert prefix == "retro", \
#             "Retro args must be prefixed with '--retro-*', for consistent " \
#             "styling. Please fix '%s'." % ", ".join(action.option_strings)

#     return parser
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
