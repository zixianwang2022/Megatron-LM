# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, and NVIDIA.
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


# Most of the code here has been copied from:
#   https://github.com/google-research/albert/blob/master/create_pretraining_data.py
# with some modifications.

import collections
import torch
import numpy as np


def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess
    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(['make', '-C', path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys
        sys.exit(1)


def build_training_sample(sample, target_seq_length,
                          max_seq_length, max_seq_length_dec,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, np_rng, bos_id=None,
                          eos_id=None, sentinel_tokens=None):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        sentinel_tokens: unique value to be substituted for every replaced span
    """

    # We assume that we have at least two sentences in the sample
    assert len(sample) > 1
    assert target_seq_length <= max_seq_length

    # Divide sample into two segments (A and B).
    # tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample, np_rng)
    tokens = get_document(sample)

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = truncate_segments(tokens, len(tokens), max_num_tokens, np_rng)

    # Build tokens and toketypes.
    tokens, tokentypes = create_tokens_and_tokentypes(tokens, cls_id, sep_id)

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng,
        max_ngrams=10, geometric_dist=True, masking_style="t5")

    # Padding.
    tokens_enc, tokentypes_enc, tokens_dec_in, labels, enc_mask, \
    dec_mask, enc_dec_mask, loss_mask \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length,
                                   max_seq_length_dec, masked_spans,
                                   bos_id, eos_id, sentinel_tokens)

    train_sample = {
        'text_enc': tokens_enc,
        'text_dec': tokens_dec_in,
        'types': tokentypes_enc,
        'labels': labels,
        'loss_mask': loss_mask,
        'truncated': int(truncated),
        'enc_mask': enc_mask,
        'dec_mask': dec_mask,
        'enc_dec_mask': enc_dec_mask,
    }
    return train_sample


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def get_document(sample):
    """
    Merge the sentences in the sample to form a document
    :param sample:
    :return:
    """
    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    doc = []
    for sent in sample:
        doc.extend(sent)

    return doc


def truncate_segments(tokens_a, len_a, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    #print(len_a, len_b, max_num_tokens)
    assert len_a > 0
    if len_a <= max_num_tokens:
        return False
    while len_a > max_num_tokens:
        len_a -= 1
        tokens = tokens_a

        if np_rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
    return True


def create_tokens_and_tokentypes(tokens_a, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build tokentypes."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)

    return tokens, tokentypes


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False,
                                 geometric_dist=False,
                                 masking_style="bert"):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                not is_start_piece(vocab_id_to_token_dict[token])):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1. / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(ngrams[:len(cand_index_set)],
                              p=pvals[:len(cand_index_set)] /
                              pvals[:len(cand_index_set)].sum(keepdims=True))
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_spans.append(MaskedLmInstance(
            index=index_set,
            label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id,
                             max_seq_length, max_seq_length_dec,
                             masked_spans=None, bos_id=None,
                             eos_id=None, sentinel_tokens=None):
    """Pad sequences and convert them to numpy."""

    sentinel_tokens = collections.deque(sentinel_tokens)
    (t5_input, t5_tokentypes) = ([], [])
    (t5_decoder_in, t5_decoder_out) = ([bos_id], [])
    (start_index, end_index) = (0, None)
    for span in masked_spans:
        flag = sentinel_tokens.popleft()

        # Append the same tokens in decoder input and output
        t5_decoder_in.append(flag)
        t5_decoder_in.extend(span.label)
        t5_decoder_out.append(flag)
        t5_decoder_out.extend(span.label)

        end_index = span.index[0]
        t5_input.extend(tokens[start_index: end_index])
        t5_input.append(flag)

        t5_tokentypes.extend(tokentypes[start_index: end_index])
        # The type of the sentinel token is the type of the end_index token
        t5_tokentypes.append(tokentypes[end_index])

        # the next start index is the token after the last span token
        start_index = span.index[-1] + 1

    # Add <eos> token to the t5_decoder_out
    t5_decoder_out.append(eos_id)

    # Add the remaining tokens to the t5 input
    t5_input.extend(tokens[start_index:])
    t5_tokentypes.extend(tokentypes[start_index:])

    # assert (len(t5_input) - len(masked_spans)) + \
    #        (len(t5_decoder_in) - (len(masked_spans) + 1)) == len(tokens)

    # Some checks.

    # Encoder-side padding mask.
    num_tokens = len(t5_input)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(t5_tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_enc = np.array(t5_input + filler, dtype=np.int64)
    tokentypes_enc = np.array(t5_tokentypes + filler, dtype=np.int64)

    # Decoder-side padding mask.
    num_tokens_dec = len(t5_decoder_in)
    padding_length_dec = max_seq_length_dec - num_tokens_dec
    assert padding_length_dec >= 0
    filler_dec = [pad_id] * padding_length_dec
    tokens_dec_in = np.array(t5_decoder_in + filler_dec, dtype=np.int64)

    # Create attention masks
    enc_mask = make_attention_mask(tokens_enc, tokens_enc)
    enc_dec_mask = make_attention_mask(tokens_dec_in, tokens_enc)
    dec_mask = make_attention_mask(tokens_dec_in, tokens_dec_in)
    dec_mask = dec_mask * make_history_mask(tokens_dec_in)

    # Labels mask.
    labels = t5_decoder_out + ([-1] * padding_length_dec)
    labels = np.array(labels, dtype=np.int64)

    # Loss mask
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    return tokens_enc, tokentypes_enc, tokens_dec_in, labels, enc_mask, \
           dec_mask, enc_dec_mask, loss_mask


def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask


def make_attention_mask_3d(source_block, target_block):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[:, None, :] >= 1) * (source_block[:, :, None] >= 1)
    # (batch, source_length, target_length)
    # mask = mask.astype(np.int64)
    return mask


def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (arange[None, ] <= arange[:, None])
    history_mask = history_mask.astype(np.int64)
    return history_mask


def make_history_mask_3d(block):
    batch, length = block.shape
    arange = torch.arange(length, device=block.device)
    history_mask = (arange[None, ] <= arange[:, None])[None, ]
    history_mask = history_mask.expand(batch, length, length)
    return history_mask
