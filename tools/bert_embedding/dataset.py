# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch

from megatron import get_args, get_tokenizer
# >>>
# from megatron.data.bert_dataset import build_training_sample
# from megatron.core.datasets.bert_dataset import build_training_sample
from megatron.core import parallel_state
from megatron.core.datasets.bert_dataset import BERTMaskedWordPieceDataset, BERTMaskedWordPieceDatasetConfig
from megatron.core.datasets.utils import Split
# <<<

# >>>
from lutil import pax
# <<<


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# class BertEmbeddingDataset(torch.utils.data.Dataset):
#     '''Dataset to convert a text dataset to Bert tokens.'''

#     def __init__(self, text_dataset, max_seq_length):

#         super().__init__()

#         args = get_args()

#         # Dataset, tokenizer.
#         self.text_dataset = text_dataset
#         self.bert_tokenizer = get_tokenizer()

#         # Params to store.
#         self.max_seq_length = max_seq_length
#         self.seed = args.seed
#         self.masked_lm_prob = args.mask_prob

#         # Vocab stuff.
#         self.vocab_id_list = list(self.bert_tokenizer.inv_vocab.keys())
#         self.vocab_id_to_token_dict = self.bert_tokenizer.inv_vocab
#         # >>>
#         # self.cls_id = self.bert_tokenizer.cls
#         # self.sep_id = self.bert_tokenizer.sep
#         # self.mask_id = self.bert_tokenizer.mask
#         # self.pad_id = self.bert_tokenizer.pad
#         # <<<

#     def __len__(self):
#         return len(self.text_dataset)

#     def __getitem__(self, idx):

#         # Text.
#         text_sample = self.text_dataset[idx]
#         text = text_sample["text"]
#         text = text.replace("<|endoftext|>", "")

#         # Bert/Wordpiece tokens (+truncate).
#         bert_token_ids = self.bert_tokenizer.tokenize(text)
#         bert_token_ids = bert_token_ids[:self.max_seq_length - 2] # cls+sep.
#         if not bert_token_ids:
#             bert_token_ids = [ self.bert_tokenizer.pad_id ] # hack when empty seq

#         # Note that this rng state should be numpy and not python since
#         # python randint is inclusive whereas the numpy one is exclusive.
#         # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
#         np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))

#         # Build sample.
#         # >>>
#         # sample = build_training_sample([bert_token_ids],
#         #                                len(bert_token_ids),
#         #                                len(bert_token_ids) + 2, # for cls+sep
#         #                                self.vocab_id_list,
#         #                                self.vocab_id_to_token_dict,
#         #                                self.cls_id, self.sep_id,
#         #                                self.mask_id, self.pad_id,
#         #                                self.masked_lm_prob, np_rng,
#         #                                binary_head=False)
#         # +++
#         sample = BERTMaskedWordPieceDataset.build_sample(
#             sample=[bert_token_ids],
#             target_sequence_length=len(bert_token_ids),
#             max_sequence_length=len(bert_token_ids) + 2, # for cls+sep
#             vocab_id_list=self.vocab_id_list,
#             vocab_id_to_token_dict=self.vocab_id_to_token_dict,
#             # >>>
#             # cls_id=self.cls_id,
#             # sep_id=self.sep_id,
#             # mask_id=self.mask_id,
#             # pad_id=self.pad_id,
#             tokenizer=self.bert_tokenizer,
#             # <<<
#             masked_lm_prob=self.masked_lm_prob,
#             np_rng=np_rng,
#             classification_head=None)
#         # <<<

#     # >>>
#     # def build_sample(cls,
#     #                  sample,
#     #                  target_sequence_length, max_seq_length,
#     #                  vocab_id_list, vocab_id_to_token_dict,
#     #                  cls_id, sep_id, mask_id, pad_id,
#     #                  masked_lm_prob, np_rng, classification_head):
#     #     # <<<
#     #     # >>>
#     #     pax("sample")
#     #     # <<<
#     #     sample["seq_length"] = len(sample["text"])
#     #     return sample
#     # <<<
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# class BertInMemoryDatasetConfig(BERTMaskedWordPieceDatasetConfig):

# class BertInMemoryDataset(BERTMaskedWordPieceDataset):

#     def __init__(self):
# class TextToBertDataset(BERTMaskedWordPieceDataset):

#     def __getitem__(self, idx):

#         pax("idx")

# class InMemoryBertDataset(BERTMaskedWordPieceDataset):

#     def _build_sample_index(
#         self, sequence_length: int, min_sentences_per_sample: int
#     ) -> np.ndarray:
#         return None

#     def __getitem__(self, idx):

#         raise Exception("hi.")

class DummySampleIndex:

    def __init__(self):
        self.n = None

    def update(self, n):
        self.n = n

    def __getitem__(self, _):
        return 0, self.n, self.n

class DummyIndexedDataset:

    def __init__(self):
        self.token_ids = None

    def update(self, token_ids):
        self.token_ids = token_ids

    def __getitem__(self, _):
        # return self.token_ids, (None, None)
        return self.token_ids

# class BertSampleBuilder:
class BertSampleBuilder(BERTMaskedWordPieceDataset):

    def __init__(self):

        args = get_args()

        tokenizer = get_tokenizer()

        pax({"seq_length": args.seq_length})

        config = BERTMaskedWordPieceDatasetConfig(
            is_built_on_rank=lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=args.data_path,
            blend_per_split=[
                args.train_data_path,
                args.valid_data_path,
                args.test_data_path,
            ],
            split=args.split,
            path_to_cache=args.data_cache_path,
            mock=False,
            tokenizer=tokenizer,
            masking_probability=args.mask_prob,
            short_sequence_probability=args.short_seq_prob,
            masking_max_ngram=3,
            masking_do_full_word=True,
            masking_do_permutation=False,
            masking_use_longer_ngrams=False,
            masking_use_geometric_distribution=False,
            # >>>
            # classification_head=args.bert_binary_head,
            # classification_head=None,
            classification_head=False,
            # <<<
        )

        # print_rank_0('> building train, validation, and test datasets '
        #              'for BERT ...')

        # train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        #     # BERTMaskedWordPieceDataset,
        #     BertInMemoryDataset,
        #     train_val_test_num_samples,
        #     config,
        # ).build()
        # self.bert_dataset = BERTMaskedWordPieceDataset(
        # self.bert_dataset = TextToBertDataset(
        # self.bert_dataset = InMemoryBertDataset(
        super().__init__(
            # indexed_dataset: MMapIndexedDataset,
            # dataset_path: str,
            # indexed_indices: numpy.ndarray,
            # num_samples: int,
            # index_split: Split,
            # config: BERTMaskedWordPieceDatasetConfig,

            indexed_dataset = DummyIndexedDataset(), # None, # InMemoryIndexedDataset(),
            dataset_path = None,
            indexed_indices = None,
            num_samples = None,
            index_split = Split.train, # None
            config = config,
        )

        # pax("dataset")

        # print_rank_0("> finished creating BERT datasets ...")

        # return train_ds, valid_ds, test_ds

    def _build_sample_index(
        self, sequence_length: int, min_sentences_per_sample: int
    ) -> np.ndarray:
        # return None
        return DummySampleIndex()

    def build(self, token_ids):
        self.sample_index.update(len(token_ids))
        self.dataset.update(token_ids)
        sample = self[0]
        # pax("token_ids, sample")
        return sample


class BertEmbeddingDataset(torch.utils.data.Dataset):
# class BertEmbeddingDataset(BERTMaskedWordPieceDataset):
    '''Dataset to convert a text dataset to Bert tokens.'''

    # def __init__(self, text_dataset, max_seq_length):

    #     super().__init__()

    #     args = get_args()

    #     # Dataset, tokenizer.
    #     self.text_dataset = text_dataset
    #     self.bert_tokenizer = get_tokenizer()

    #     # Params to store.
    #     self.max_seq_length = max_seq_length
    #     self.seed = args.seed
    #     self.masked_lm_prob = args.mask_prob

    #     # Vocab stuff.
    #     self.vocab_id_list = list(self.bert_tokenizer.inv_vocab.keys())
    #     self.vocab_id_to_token_dict = self.bert_tokenizer.inv_vocab
    #     # >>>
    #     # self.cls_id = self.bert_tokenizer.cls
    #     # self.sep_id = self.bert_tokenizer.sep
    #     # self.mask_id = self.bert_tokenizer.mask
    #     # self.pad_id = self.bert_tokenizer.pad
    #     # <<<
    def __init__(self, text_dataset, max_seq_length):

        super().__init__()

        args = get_args()

        pax("max_seq_length", {"seq_length": args.seq_length})

        # Dataset, tokenizer.
        self.text_dataset = text_dataset
        self.max_seq_length = max_seq_length
        self.bert_tokenizer = get_tokenizer()
        self.bert_sample_builder = BertSampleBuilder()

        # # Params to store.
        # self.max_seq_length = max_seq_length
        # self.seed = args.seed
        # self.masked_lm_prob = args.mask_prob

        # # Vocab stuff.
        # self.vocab_id_list = list(self.bert_tokenizer.inv_vocab.keys())
        # self.vocab_id_to_token_dict = self.bert_tokenizer.inv_vocab
        # # >>>
        # # self.cls_id = self.bert_tokenizer.cls
        # # self.sep_id = self.bert_tokenizer.sep
        # # self.mask_id = self.bert_tokenizer.mask
        # # self.pad_id = self.bert_tokenizer.pad
        # # <<<

    def __len__(self):
        return len(self.text_dataset)

    # >>>
    # def __getitem__(self, idx):

    #     # Text.
    #     text_sample = self.text_dataset[idx]
    #     text = text_sample["text"]
    #     text = text.replace("<|endoftext|>", "")

    #     # Bert/Wordpiece tokens (+truncate).
    #     bert_token_ids = self.bert_tokenizer.tokenize(text)
    #     bert_token_ids = bert_token_ids[:self.max_seq_length - 2] # cls+sep.
    #     if not bert_token_ids:
    #         bert_token_ids = [ self.bert_tokenizer.pad_id ] # hack when empty seq

    #     # Note that this rng state should be numpy and not python since
    #     # python randint is inclusive whereas the numpy one is exclusive.
    #     # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
    #     np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))

    #     # Build sample.
    #     # >>>
    #     # sample = build_training_sample([bert_token_ids],
    #     #                                len(bert_token_ids),
    #     #                                len(bert_token_ids) + 2, # for cls+sep
    #     #                                self.vocab_id_list,
    #     #                                self.vocab_id_to_token_dict,
    #     #                                self.cls_id, self.sep_id,
    #     #                                self.mask_id, self.pad_id,
    #     #                                self.masked_lm_prob, np_rng,
    #     #                                binary_head=False)
    #     # +++
    #     sample = BERTMaskedWordPieceDataset.build_sample(
    #         sample=[bert_token_ids],
    #         target_sequence_length=len(bert_token_ids),
    #         max_sequence_length=len(bert_token_ids) + 2, # for cls+sep
    #         vocab_id_list=self.vocab_id_list,
    #         vocab_id_to_token_dict=self.vocab_id_to_token_dict,
    #         # >>>
    #         # cls_id=self.cls_id,
    #         # sep_id=self.sep_id,
    #         # mask_id=self.mask_id,
    #         # pad_id=self.pad_id,
    #         tokenizer=self.bert_tokenizer,
    #         # <<<
    #         masked_lm_prob=self.masked_lm_prob,
    #         np_rng=np_rng,
    #         classification_head=None)
    #     # <<<
    def __getitem__(self, idx):

        # Text.
        text_sample = self.text_dataset[idx]
        text = text_sample["text"]
        text = text.replace("<|endoftext|>", "")

        # Bert/Wordpiece tokens (+truncate).
        bert_token_ids = self.bert_tokenizer.tokenize(text)
        bert_token_ids = bert_token_ids[:self.max_seq_length - 2] # cls+sep.
        if not bert_token_ids:
            bert_token_ids = [ self.bert_tokenizer.pad_id ] # hack when empty seq

        # Bert sample.
        sample = self.bert_sample_builder.build(bert_token_ids)

        # >>>
        # pax("sample")
        # <<<

        return sample

    # <<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
