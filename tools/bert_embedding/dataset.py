# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# import numpy as np
# import torch

# from megatron import get_args, get_tokenizer
# >>>
# from megatron.data.bert_dataset import build_training_sample
# from megatron.core.datasets.bert_dataset import build_training_sample
# from megatron.core import parallel_state
# from megatron.core.datasets.bert_dataset import BERTMaskedWordPieceDataset, BERTMaskedWordPieceDatasetConfig
# from megatron.core.datasets.utils import Split
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

# class DummySampleIndex:

#     def __init__(self):
#         self.n = None

#     def update(self, n):
#         self.n = n

#     def __getitem__(self, _):
#         return 0, self.n, self.n

# class DummyIndexedDataset:

#     def __init__(self):
#         self.token_ids = None

#     def update(self, token_ids):
#         self.token_ids = token_ids

#     def __getitem__(self, _):
#         # return self.token_ids, (None, None)
#         return self.token_ids

# # class BertSampleBuilder:
# class BertSampleBuilder(BERTMaskedWordPieceDataset):

#     def __init__(self):

#         args = get_args()

#         tokenizer = get_tokenizer()

#         # pax({"seq_length": args.seq_length})

#         # sample = BERTMaskedWordPieceDataset.build_sample(
#         #     sample=[bert_token_ids],
#         #     target_sequence_length=len(bert_token_ids),
#         #     max_sequence_length=len(bert_token_ids) + 2, # for cls+sep
#         #     vocab_id_list=self.vocab_id_list,
#         #     vocab_id_to_token_dict=self.vocab_id_to_token_dict,
#         #     tokenizer=self.bert_tokenizer,
#         #     masked_lm_prob=self.masked_lm_prob,
#         #     np_rng=np_rng,
#         #     classification_head=None)
#         config = BERTMaskedWordPieceDatasetConfig(
#             is_built_on_rank=lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
#             random_seed=args.seed,
#             # >>>
#             # sequence_length=args.seq_length,
#             # sequence_length=None,
#             sequence_length=-1,
#             # <<<
#             blend=args.data_path,
#             blend_per_split=[
#                 args.train_data_path,
#                 args.valid_data_path,
#                 args.test_data_path,
#             ],
#             split=args.split,
#             path_to_cache=args.data_cache_path,
#             mock=False,
#             tokenizer=tokenizer,
#             masking_probability=args.mask_prob,
#             short_sequence_probability=args.short_seq_prob,
#             masking_max_ngram=3,
#             masking_do_full_word=True,
#             masking_do_permutation=False,
#             masking_use_longer_ngrams=False,
#             masking_use_geometric_distribution=False,
#             # >>>
#             # classification_head=args.bert_binary_head,
#             # classification_head=None,
#             classification_head=False,
#             # <<<
#         )

#         # print_rank_0('> building train, validation, and test datasets '
#         #              'for BERT ...')

#         # train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
#         #     # BERTMaskedWordPieceDataset,
#         #     BertInMemoryDataset,
#         #     train_val_test_num_samples,
#         #     config,
#         # ).build()
#         # self.bert_dataset = BERTMaskedWordPieceDataset(
#         # self.bert_dataset = TextToBertDataset(
#         # self.bert_dataset = InMemoryBertDataset(
#         super().__init__(
#             # indexed_dataset: MMapIndexedDataset,
#             # dataset_path: str,
#             # indexed_indices: numpy.ndarray,
#             # num_samples: int,
#             # index_split: Split,
#             # config: BERTMaskedWordPieceDatasetConfig,

#             indexed_dataset = DummyIndexedDataset(), # None, # InMemoryIndexedDataset(),
#             dataset_path = None,
#             indexed_indices = None,
#             num_samples = None,
#             index_split = Split.train, # None
#             config = config,
#         )

#         # pax("dataset")

#         # print_rank_0("> finished creating BERT datasets ...")

#         # return train_ds, valid_ds, test_ds

#     def _build_sample_index(
#         self, sequence_length: int, min_sentences_per_sample: int
#     ) -> np.ndarray:
#         # return None
#         return DummySampleIndex()

#     def build(self, token_ids):
#         # >>>
#         # self.config.sequence_length = len(token_ids) + 2 # for cls, sep
#         self.config.sequence_length = len(token_ids) + 2 # for cls, sep
#         # self.config.sequence_length = None
#         # <<<
#         self.sample_index.update(len(token_ids))
#         self.dataset.update(token_ids)
#         sample = self[0]
#         # pax("token_ids, sample")
#         return sample


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

        # pax("max_seq_length", {"seq_length": args.seq_length})

        # Dataset, tokenizer.
        self.text_dataset = text_dataset
        self.max_seq_length = max_seq_length
        self.bert_tokenizer = get_tokenizer()
        # self.bert_sample_builder = BertSampleBuilder()
        # self.bert_sample_builder = BertSampleBuilder(max_seq_length)

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
        # sample = self.bert_sample_builder.build(bert_token_ids)
        # sample = build_old_sample(self.bert_tokenizer, idx, bert_token_ids)
        sample = build_fast_sample(self.bert_tokenizer, bert_token_ids)

        # >>>
        # # if True:
        # if len(bert_token_ids) > 150: # 254:
        # # if len(bert_token_ids) > 50: # 254:
        #     old_sample = build_old_sample(self.bert_tokenizer, idx, bert_token_ids)
        #     fast_sample = build_fast_sample(self.bert_tokenizer, bert_token_ids)
        #     # pax("text, bert_token_ids, sample, old_sample, fast_sample")

        #     pax({
        #         "tokens" : list(zip(
        #             fast_sample["text"],
        #             sample["text"],
        #             old_sample["text"],
        #             [ int(a == b) for a, b in
        #               zip(fast_sample["text"], sample["text"]) ],
        #             [ int(a == b) for a, b in
        #               zip(fast_sample["text"], old_sample["text"]) ],
        #         )),
        #         "diff" : "%d / %d" % (
        #             sum(a!=b for a,b in zip(fast_sample["text"],sample["text"])),
        #             len(fast_sample["text"]),
        #         ),
        #         "old diff" : "%d / %d" % (
        #             sum(a!=b for a,b in zip(fast_sample["text"],old_sample["text"])),
        #             len(fast_sample["text"]),
        #         ),
        #         "mask_prob" : get_args().mask_prob,
        #     })
        # <<<

        return sample
    # <<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def build_fast_sample(tokenizer, token_ids):
    get_constant_array = lambda c : np.full((len(token_ids) + 2,), c, "int64")
    return {
        "text" : np.array([ tokenizer.cls, *token_ids, tokenizer.sep ], dtype="int64"),
        "types" : get_constant_array(0),
        "labels" : get_constant_array(-1),
        "is_random" : 0,
        "loss_mask" : get_constant_array(0),
        "padding_mask" : get_constant_array(1),
        "truncated" : 0,
    }
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# from megatron.data.dataset_utils import (
#     get_samples_mapping,
#     get_a_and_b_segments,
#     truncate_segments,
#     create_tokens_and_tokentypes,
#     create_masked_lm_predictions
# )


# def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
#                              masked_labels, pad_id, max_seq_length):
#     """Pad sequences and convert them to numpy."""

#     # Some checks.
#     num_tokens = len(tokens)
#     padding_length = max_seq_length - num_tokens
#     assert padding_length >= 0, \
#         f"num_tokens ({num_tokens}) is greater than " \
#         "max_seq_length ({max_seq_length})."
#     assert len(tokentypes) == num_tokens
#     assert len(masked_positions) == len(masked_labels)

#     # Tokens and token types.
#     filler = [pad_id] * padding_length
#     tokens_np = np.array(tokens + filler, dtype=np.int64)
#     tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

#     # Padding mask.
#     padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
#                                dtype=np.int64)

#     # Lables and loss mask.
#     labels = [-1] * max_seq_length
#     loss_mask = [0] * max_seq_length
#     for i in range(len(masked_positions)):
#         assert masked_positions[i] < num_tokens
#         labels[masked_positions[i]] = masked_labels[i]
#         loss_mask[masked_positions[i]] = 1
#     labels_np = np.array(labels, dtype=np.int64)
#     loss_mask_np = np.array(loss_mask, dtype=np.int64)

#     return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


# def build_training_sample(sample,
#                           target_seq_length, max_seq_length,
#                           vocab_id_list, vocab_id_to_token_dict,
#                           cls_id, sep_id, mask_id, pad_id,
#                           masked_lm_prob, np_rng, binary_head):
#     """Biuld training sample.

#     Arguments:
#         sample: A list of sentences in which each sentence is a list token ids.
#         target_seq_length: Desired sequence length.
#         max_seq_length: Maximum length of the sequence. All values are padded to
#             this length.
#         vocab_id_list: List of vocabulary ids. Used to pick a random id.
#         vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
#         cls_id: Start of example id.
#         sep_id: Separator id.
#         mask_id: Mask token id.
#         pad_id: Padding token id.
#         masked_lm_prob: Probability to mask tokens.
#         np_rng: Random number genenrator. Note that this rng state should be
#               numpy and not python since python randint is inclusive for
#               the opper bound whereas the numpy one is exclusive.
#     """

#     if binary_head:
#         # We assume that we have at least two sentences in the sample
#         assert len(sample) > 1
#     assert target_seq_length <= max_seq_length

#     # Divide sample into two segments (A and B).
#     if binary_head:
#         tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample,
#                                                                   np_rng)
#     else:
#         tokens_a = []
#         for j in range(len(sample)):
#             tokens_a.extend(sample[j])
#         tokens_b = []
#         is_next_random = False

#     # Truncate to `target_sequence_length`.
#     max_num_tokens = target_seq_length
#     truncated = truncate_segments(tokens_a, tokens_b, len(tokens_a),
#                                   len(tokens_b), max_num_tokens, np_rng)

#     # Build tokens and toketypes.
#     tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b,
#                                                       cls_id, sep_id)

#     # Masking.
#     max_predictions_per_seq = masked_lm_prob * max_num_tokens
#     # >>>
#     # old_tokens = tokens
#     # <<<
#     (tokens, masked_positions, masked_labels, _, _) = create_masked_lm_predictions(
#         tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
#         cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)
#     # >>>
#     # pax({
#     #     "old tokens" : list(zip(old_tokens, tokens, [int(a==b) for a,b in zip(old_tokens, tokens)])),
#     #     "diff" : "%d / %d" % (
#     #         sum(a != b for a, b in zip(old_tokens, tokens)),
#     #         len(old_tokens),
#     #     ),
#     # }, "masked_lm_prob")
#     # <<<

#     # Padding.
#     tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
#         = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
#                                    masked_labels, pad_id, max_seq_length)

#     train_sample = {
#         'text': tokens_np,
#         'types': tokentypes_np,
#         'labels': labels_np,
#         'is_random': int(is_next_random),
#         'loss_mask': loss_mask_np,
#         'padding_mask': padding_mask_np,
#         'truncated': int(truncated)}
#     return train_sample


# def build_old_sample(tokenizer, idx, bert_token_ids):
#     args = get_args()
#     np_rng = np.random.RandomState(seed=((args.seed + idx) % 2**32))
#     vocab_id_list = list(tokenizer.inv_vocab.keys())
#     vocab_id_to_token_dict = tokenizer.inv_vocab
#     sample = build_training_sample([bert_token_ids],
#                                    len(bert_token_ids),
#                                    len(bert_token_ids) + 2, # for cls+sep
#                                    vocab_id_list,
#                                    vocab_id_to_token_dict,
#                                    tokenizer.cls_id, tokenizer.sep_id,
#                                    tokenizer.mask_id, tokenizer.pad_id,
#                                    args.mask_prob, np_rng,
#                                    binary_head=False)

#     # pax("sample")
#     return sample
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
