# # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

? ? ?

# from megatron.core.models.retro.data.query import (
#     # >>>
#     multi_split_gpt_train_valid_test_datasets_provider,
#     # <<<
# )
# # >>>
# from megatron.core.models.retro.data.query.multi_split_gpt_dataset import \
#     core_multi_split_gpt_dataset_config_from_retro_preprocessing_config
# # <<<
# # >>>
# # from megatron.training import (
# #     build_train_valid_test_datasets,
# #     get_train_valid_test_num_samples,
# #     update_train_iters,
# # )
# from pretrain_gpt import is_dataset_built_on_rank
# # <<<

# ? ? ?

# def get_gpt_datasets(config, split, return_document_ids, custom_data_path):

#     # Dataset config.
#     data_config = core_multi_split_gpt_dataset_config_from_retro_preprocessing_config(
#         config=config,
#         split=split,
#         return_document_ids=return_document_ids,
#         is_dataset_built_on_rank=is_dataset_built_on_rank,
#         custom_data_path=custom_data_path,
#     )

#     # Datasets.
#     print_rank_0(" > datasets.")
#     train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
#         lambda n : train_valid_test_datasets_provider(data_config, n))

#     num_train_samples, num_valid_samples, num_test_samples = \
#         get_train_valid_test_num_samples()

#     datasets = RetroGPTDatasets(
#         train=(train_ds, num_train_samples),
#         valid=(valid_ds, num_valid_samples),
#         test=(test_ds, num_test_samples),
#     )

#     return datasets
