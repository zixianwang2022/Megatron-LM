# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

""" Tasks data utility."""

import re
import numpy as np

from megatron.data.t5_dataset_utils import make_attention_mask, make_history_mask

def clean_text(text):
    """Remove new lines and multiple spaces and adjust end of sentence dot."""

    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    for _ in range(3):
        text = text.replace(' . ', '. ')

    return text


def build_sample(enc_ids, tokentypes_enc,
                 dec_in_ids, labels, loss_mask, references=[]):
    """Convert to numpy and return a sample consumed by the batch producer."""
    """References are always empty except SQuAD 1.1"""

    enc_ids = np.array(enc_ids, dtype=np.int64)
    dec_in_ids = np.array(dec_in_ids, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)
    tokentypes_enc = np.array(tokentypes_enc, dtype=np.int64)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    enc_mask = make_attention_mask(enc_ids, enc_ids)
    enc_dec_mask = make_attention_mask(dec_in_ids, enc_ids)
    dec_mask = make_attention_mask(dec_in_ids, dec_in_ids)
    dec_mask = dec_mask * make_history_mask(dec_in_ids)

    sample = {
        'text_enc': enc_ids,
        'text_dec': dec_in_ids,
        'types': tokentypes_enc,
        'labels': labels,
        'loss_mask': loss_mask,
        'enc_mask': enc_mask,
        'dec_mask': dec_mask,
        'enc_dec_mask': enc_dec_mask,
        'references': references
    }

    return sample


def build_tokens_types_paddings_from_text(src_text, trg_text,
                                          tokenizer, max_seq_length,
                                          decoder_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    src_text_ids = tokenizer.tokenize(src_text)
    trg_text_ids = None
    if trg_text is not None:
        trg_text_ids = tokenizer.tokenize(trg_text)

    return build_tokens_types_paddings_from_ids(src_text_ids, trg_text_ids,
                                                max_seq_length, decoder_seq_length,
                                                tokenizer.cls, tokenizer.sep,
                                                tokenizer.pad,
                                                tokenizer.bos_token_id,
                                                tokenizer.eos_token_id)


def mnli_build_tokens_types_paddings_from_text(sentence1, sentence2, label,
                                               tokenizer, max_seq_length,
                                               decoder_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    src_text = "sentence1: " + sentence1 + " " + "sentence2: " + sentence2
    src_text_ids = tokenizer.tokenize(src_text)
    trg_text_ids = None
    if label is not None:
        trg_text_ids = tokenizer.tokenize(label)

    return build_tokens_types_paddings_from_ids(src_text_ids, trg_text_ids,
                                                max_seq_length, decoder_seq_length,
                                                tokenizer.cls, tokenizer.sep,
                                                tokenizer.pad,
                                                tokenizer.bos_token_id,
                                                tokenizer.eos_token_id)


def build_tokens_types_paddings_from_ids(src_ids, trg_ids, max_seq_length,
                                         decoder_seq_length,
                                         cls_id, sep_id, pad_id,
                                         bos_id=None, eos_id=None):
    """Build token types and paddings, trim if needed, and pad if needed."""

    enc_ids = []
    tokentypes_enc = []

    # [CLS].
    enc_ids.append(cls_id)
    tokentypes_enc.append(0)

    # A.
    len_src = len(src_ids)
    enc_ids.extend(src_ids)
    tokentypes_enc.extend([0] * len_src)

    # Cap the size.
    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]
        tokentypes_enc = tokentypes_enc[0: max_seq_length - 1]

    # [SEP].
    enc_ids.append(sep_id)
    tokentypes_enc.append(0)

    # Padding.
    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)
        tokentypes_enc.extend([pad_id] * padding_length)

    # B.
    # Not enforcing sequence length checking for target sequences
    # if trg_ids is not None:
    dec_in_ids, dec_out_ids = [bos_id], []
    dec_in_ids.extend(trg_ids)
    dec_out_ids.extend(trg_ids)

    if len(dec_in_ids) > decoder_seq_length:
        dec_in_ids = dec_in_ids[0: decoder_seq_length]
        dec_out_ids = dec_out_ids[0: decoder_seq_length - 1]

    dec_out_ids.append(eos_id)

    # Decoder-side padding mask.
    num_tokens_dec = len(dec_in_ids)
    padding_length_dec = decoder_seq_length - num_tokens_dec
    assert padding_length_dec >= 0

    filler_dec = [pad_id] * padding_length_dec
    dec_in_ids.extend(filler_dec)
    dec_out_ids.extend(filler_dec)

    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)

    return enc_ids, tokentypes_enc, dec_in_ids, \
           dec_out_ids, loss_mask
