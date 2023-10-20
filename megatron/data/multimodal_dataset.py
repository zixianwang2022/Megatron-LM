# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import hashlib
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import torch
import yaml
import json
from random import randrange
from megatron import get_args
import os
from megatron import print_rank_0
import time
from megatron.core import mpu

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import Compose, ToPILImage, RandAugment, RandomResizedCrop, Resize

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_train(img_h, img_w):
    return Compose([
        ToPILImage(),
        RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0)),
        _convert_image_to_rgb,
    ])

def _transform_train_aug(img_h, img_w):
    return Compose([
        ToPILImage(),
        RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0)),
        _convert_image_to_rgb,
        RandAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    ])

def _transform_test(img_h, img_w):
    return Compose([
        ToPILImage(),
        Resize((img_h, img_w)),
        _convert_image_to_rgb,
    ])

pixel_mean = [123.675, 116.28, 103.53]
pixel_std = [58.395, 57.12, 57.375]
# # clip preprocessing
# clip_pixel_mean = [122.7709383 , 116.7460125 , 104.09373615]
# clip_pixel_std = [68.5005327 ,  66.6321579 ,  70.32316305]
# clip preprocessing
clip_pixel_mean = [123.675, 116.28, 103.53]
clip_pixel_std = [58.395, 57.12, 57.375]

def _build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    desc = "Multimodal Dataset\n\n"
    desc += f"Data prefix {data_prefix}\n"
    desc += f"Dataset name {name}\n"
    desc += f"Number of samples {num_samples}\n"
    desc += f"Sequence length {seq_length}\n"
    desc += f"Random seed {seed}\n"
    desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()

    _filename = data_prefix + data_prefix.split("/")[-1]
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting '
                      'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - \
                                         num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
                    'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = (last_epoch_num_samples <
                                       int(0.80 * num_samples_per_epoch))
                if separate_last_epoch:
                    string = ' > last epoch number of samples ({}) is smaller '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to True'
                else:
                    string = ' > last epoch number of samples ({}) is larger '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to False'
                print(string.format(last_epoch_num_samples,
                                    num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                               num_epochs, tokens_per_epoch)

            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx, desc, desc_hash

def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length,
                      num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))

class MultiModalDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, indexed_dataset,
                 num_samples, seq_length, seed, visual_indexed_dataset=None):
        args = get_args()
        self.args = args
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.visual_indexed_dataset = visual_indexed_dataset
        #self.doc_idx = indexed_dataset.get_doc_idx()
        self.data_prefix = data_prefix

        dset_name = self.data_prefix.split("/")[-1]
        dset_config = yaml.safe_load(open(args.dset_config))[dset_name]

        self.aug = dset_config["augmentation"]
        self.dataset_type = dset_config["category"]

        self.prompts = json.load(open(args.prompt_path))[self.dataset_type]["tokenized"]
        self.seq_len = seq_length
        self.num_samples = num_samples

        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.debug = args.debug_log

        self.doc_idx, self.sample_idx, self.shuffle_idx, self.desc, self.desc_hash = _build_index_mappings(
            self.name, data_prefix, np.arange(len(indexed_dataset)), self.indexed_dataset.sizes,
            num_samples, seq_length, seed)

    def __len__(self):
        return self.num_samples
        #return self.indexed_dataset.sizes.shape[0]

    def __getitem__(self, idx):
        data_item = None
        while True:
            try:
                data_item = self.__getitem__local(idx)
                break
            except Exception as e:
                print('bad dataset item, re-fetching...', e)
                idx = randrange(self.num_samples) # (idx + 1) % self.num_samples
        return data_item

    def __getitem__local(self, idx):
        idx = self.shuffle_idx[idx]
        idx = self.sample_idx[idx][0]

        if self.visual_indexed_dataset is not None:
            text_sample = self.indexed_dataset.get(self.doc_idx[idx])
            img_sample = self.visual_indexed_dataset.get(self.doc_idx[idx])
            img_sample = np.array(Image.open(io.BytesIO(img_sample.tobytes(order='C'))))
        else:
            text_sample = self.indexed_dataset.get(self.doc_idx[idx])
            img_sample = self.indexed_dataset.get(self.doc_idx[idx]+1)
            img_pad = img_sample[0].item()
            xs = img_sample[1:].tobytes(order='C')
            xs = xs[:len(xs)-img_pad]
            img_sample = np.array(Image.open(io.BytesIO(xs)))

        split_token = 313131
        eod_token = 3

        raw_h, raw_w = img_sample.shape[0], img_sample.shape[1]

        seq_len = self.seq_len + 4 # len(text_sample) # Hardcoded

        prompt_idx = np.random.randint(len(self.prompts))
        # print(self.prompts, prompt_idx)
        cur_prompt = np.array(self.prompts[prompt_idx])
        # print(cur_prompt)
        if self.dataset_type == "VQA" or self.dataset_type == "Embedded":
            prompt_len = np.array([len(cur_prompt) + text_sample[1]])
            indices = np.where(text_sample == split_token)[0] - 1
            # DANNY` DEBUG print(prompt_len, indices, self.data_prefix)
            prob = text_sample[indices].astype(np.float32) / text_sample[0]
            answer_idx = np.random.choice(prob.shape[0], 1, p=prob)[0]
            st = indices[answer_idx - 1] + 2 if answer_idx > 0 else int(text_sample[1])+2
            ed = indices[answer_idx]
            text_sample = np.concatenate([text_sample[2:int(text_sample[1])+2], text_sample[st:ed], text_sample[indices[-1]+2:]])
            text_sample = np.pad(text_sample, pad_width=(0,max(0,seq_len-len(text_sample))), mode='constant', constant_values=eod_token)
        else:
            prompt_len = np.array([len(cur_prompt)])

        if len(cur_prompt) > 0:
            if self.debug == False:
                text_sample = np.concatenate([cur_prompt, text_sample])[:seq_len]

        H, W = self.args.img_h, self.args.img_w

        ratio = float(max(self.args.img_h, self.args.img_w)) / max(raw_h, raw_w)
        H, W = int(raw_h * ratio + 0.5), int(raw_w * ratio + 0.5)

        if self.aug == True:
            if self.args.aug:
                transform = _transform_train_aug(H, W)
            else:
                transform = _transform_train(H, W)
        else:
            transform = _transform_test(H, W)

        img_sample = transform(img_sample)
        img_sample = (torch.Tensor(np.array(img_sample)).permute(2, 0, 1) - self.pixel_mean) / self.pixel_std

        delta_h, delta_w = self.args.img_h - H, self.args.img_w - W
        img_sample = torch.nn.functional.pad(img_sample, (0, delta_w, 0, delta_h))

        img_sample = img_sample.reshape(-1)

        return {'text': np.array(text_sample, dtype=np.int64),
                'img': np.array(img_sample, dtype=np.float32),
                'prompt_len': np.array(prompt_len, dtype=np.int64)
               }
