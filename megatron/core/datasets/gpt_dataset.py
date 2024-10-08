# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy
import torch
import random

from megatron.training import get_tokenizer
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.core.datasets.utils import Split, log_single_rank

logger = logging.getLogger(__name__)

_PAD_TOKEN_ID = -1


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core GPT datasets"""

    reset_position_ids: bool = None
    """Option to reset the position IDs in the dataset at an interval"""

    reset_attention_mask: bool = None
    """Option to reset the attention mask from the dataset"""

    eod_mask_loss: bool = None
    """Option to enable the EOD mask loss"""

    create_attention_mask: bool = True
    """Option to enable the attention masks generation. Can be disabled if attention kernel
       generates masks by itself.
    """

    drop_last_partial_validation_sequence: bool = True
    """Option to drop the last partial validation sequence"""

    add_extra_token_to_sequence: bool = True
    """Option to draw sequences with one extra token to ensure the sample input tokens and sample
       output tokens are both of the desired sequence length
    """

    def __post_init__(self) -> None:
        """Do asserts and set fields post init
        """
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.reset_position_ids is not None
        assert self.reset_attention_mask is not None
        assert self.eod_mask_loss is not None


class GPTDataset(MegatronDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        try:
            self._pad_token_id = self.config.tokenizer.pad
        except:
            self._pad_token_id = _PAD_TOKEN_ID

        (
            self.document_index,
            self.sample_index,
            self.shuffle_index,
        ) = self._build_document_sample_shuffle_indices()

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        # Zixian: Oct 7: Modified
        # return self.sample_index.shape[0] - 1
        return len(self.sample_index)

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        
        print (f'\n\n I am getting next data of the gptdataset ')
        print (f' idx is: {idx}')
        print (f' self.config.add_extra_token_to_sequence is: {self.config.add_extra_token_to_sequence}')
        
        
        # Zixian: Oct 7: Modified
        # if idx is None:
        #     # Batch padding sequence so the index does not matter
        #     text, _ = self._query_document_sample_shuffle_indices(0)
        # else:
        #     text, _ = self._query_document_sample_shuffle_indices(idx)
        if idx is None:
            # Batch padding sequence so the index does not matter
            idx = 0  # Use the first sample for padding
        text, _ = self._query_document_sample_shuffle_indices(idx)
        
        
        
            
        # print (f'text-0: {text}')
        print (f'len (text-0): {len (text)}')

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id
            
        # print (f'text-1: {text}')
        print (f'len (text-1): {len (text)}')
        
        # assert False 

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the current and next sample indices
        doc_idx, offset = self.sample_index[idx]
        next_doc_idx, next_offset = self.sample_index[idx + 1]

        # If the current and next samples are from the same document, compute length directly
        if doc_idx == next_doc_idx:
            length = next_offset - offset
        else:
            # We're at the end of the document; get the remaining length
            doc_id = self.document_index[doc_idx]
            length = self.dataset.sequence_lengths[doc_id] - offset

        # Fetch the document id
        document_id = self.document_index[doc_idx]

        # Fetch the sample from the dataset
        sample = self.dataset.get(document_id, offset=offset, length=length)
        
        
        tokenizer = get_tokenizer ()
        a = tokenizer.detokenize (sample.tolist())
        print (f'\n input_ids : a: {sample} \n')
        print (f'\n detokenized : a: {a} \n')

        # Pad the sample if necessary
        max_length = self.config.sequence_length + self.config.add_extra_token_to_sequence
        if length < max_length:
            padding = numpy.full((max_length - length,), self._pad_token_id, dtype=sample.dtype)
            sample = numpy.concatenate((sample, padding), axis=0)

        return sample, numpy.array([document_id], dtype=numpy.int64)

# -----------------------------------------------------------------------------------------------
# ----------------------- Origin Code for multi documents per 1 sequence ------------------------
# -----------------------------------------------------------------------------------------------
        # # Get the beginning and end documents and offsets
        # doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        # doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        # document_ids = []
        # sample_parts = []

        # print (f'\n\n Inside gpt_dataset.py \n')
        # print (f'\n doc_index_beg: {doc_index_beg} \n ')
        # print (f'\n doc_index_beg_offset: {doc_index_beg_offset} \n ')
        # print (f'\n doc_index_end: {doc_index_end} \n ')
        # print (f'\n doc_index_end_offset: {doc_index_end_offset} \n ')
        # # print (f'\n doc_index_beg: {doc_index_beg} \n ')
        
        # # print (f'\n document_ids : {document_ids} \n')
        # # print (f'\n self.document_index[doc_index_beg] : {self.document_index[doc_index_beg]} \n')
        # # print (f'\n doc_index_beg_offset : {doc_index_beg_offset} \n')
        # # print (f'\n doc_index_end_offset : {doc_index_end_offset} \n')
        # # print (f'\n length : {doc_index_end_offset - doc_index_beg_offset + self.config.add_extra_token_to_sequence} \n')


        # # Sample spans a single document
        # if doc_index_beg == doc_index_end:
        #     # Add the document id
        #     document_ids.append(self.document_index[doc_index_beg])
            
            
        #     # Add the entire sample
        #     sample_parts.append(
        #         self.dataset.get(
        #             self.document_index[doc_index_beg],
        #             offset=doc_index_beg_offset,
        #             length=doc_index_end_offset
        #             - doc_index_beg_offset
        #             + self.config.add_extra_token_to_sequence,
        #         )
        #     )


        
        # # Sample spans multiple documents
        # else:
        #     # Zixian: Oct 6: random sample 1 document within the possible documents to use for fine-tune
        #     # random_i = random.randint(doc_index_beg+1, doc_index_end-1)
        #     random_i = doc_index_beg+1
        #     # random_i = doc_index_end-2 
            
        #     for i in range(doc_index_beg, doc_index_end + 1):
                
        #         # if i == doc_index_beg: 
        #         # if i in [doc_index_beg, doc_index_beg+1, doc_index_beg+2]: 
        #             # Zixian: Debug: include this to skip the first data
        #             # Because the doc_index_beg_offset will start at the middle of the sequence
        #             # continue 
        #             # a1 = 1 
        #         if i != random_i:
        #             continue  # Skip all indices except the randomly selected one

                    
        #         if i == random_i+1: 
        #             # Zixian: Debug: hacky way to see if the input_ids can only contain 1 data 
        #             # Instead of a concatenation of all of them
        #             break 
                
        #         # Add the document id
        #         document_ids.append(self.document_index[i])

        #         # Add the sample part
        #         offset = 0 if i > doc_index_beg else doc_index_beg_offset
                
        #         # Zixian: Debug to see if this is causing cutting off tokens for 1st processed sample:
        #         # offset = 0 if i >= doc_index_beg else doc_index_beg_offset
                
        #         length = (
        #             None
        #             if i < doc_index_end
        #             else doc_index_end_offset + self.config.add_extra_token_to_sequence
        #         )
        #         sample_parts.append(
        #             self.dataset.get(self.document_index[i], offset=offset, length=length)
        #         )
        # assert len(document_ids) == len(
        #     sample_parts
        # ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        # length = sum(map(len, sample_parts))
        
        # # print (f'\n sample_parts[0:2]: {sample_parts[0:3]} \n')
        # # print (f'\n document_ids[0:2]: {document_ids[0:3]} \n')
        # # print (f'\n sum(map(len, sample_parts[0:2])): {sum(map(len, sample_parts[0:2]))} \n')
        
        # from megatron.training import get_tokenizer
        # tokenizer = get_tokenizer ()
        # a = tokenizer.detokenize (sample_parts[0].tolist())
        # print (f'\n input_ids : a: {sample_parts[0]} \n')
        # print (f'\n detokenized : a: {a} \n')
        # # b = tokenizer.detokenize (sample_parts[1].tolist())
        # # print (f'\n input_ids : b: {sample_parts[1]} \n')
        # # print (f'\n detokenized : b: {b} \n')
        # # c = tokenizer.detokenize (sample_parts[2].tolist())
        # # print (f'\n input_ids : c: {sample_parts[2]} \n')
        # # print (f'\n detokenized : c: {c} \n')
        
        
        
        # # Zixian: hacky way to experiment 
        # # sample_parts = sample_parts[:1]

        # # Pad the sample if necessary
        # if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            
        #     print (f'\n I am appending pad tokens! \n')
            
        #     sample_parts.append(
        #         [self._pad_token_id]
        #         * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
        #     )

        # return (
        #     numpy.concatenate(sample_parts, dtype=numpy.int64),
        #     numpy.array(document_ids, dtype=numpy.int64),
        # )
        
# -----------------------------------------------------------------------------------------------
# ----------------------- END ------------------------
# -----------------------------------------------------------------------------------------------
        
    def _custom_build_sample_idx(
        self, 
        sizes,
        doc_idx,
        seq_length,
        num_epochs,
        tokens_per_epoch,
        drop_last_partial_sequence=True,
        add_extra_token_to_sequence=1,
    ) -> numpy.ndarray:
        """
        Build sample_idx ensuring that each sample comes from only one document.

        Args:
            sizes (numpy.ndarray): Lengths of the documents.
            doc_idx (numpy.ndarray): Indices of documents to include.
            seq_length (int): Sequence length.
            num_epochs (int): Number of epochs.
            tokens_per_epoch (int): Number of tokens per epoch.
            drop_last_partial_sequence (bool): Whether to drop sequences shorter than seq_length.
            add_extra_token_to_sequence (int): Additional tokens to add to the sequence length.

        Returns:
            numpy.ndarray: Sample index array of shape [num_samples + 1, 2].
        """
        import numpy as np

        # Consistency checks.
        assert seq_length > 1
        assert num_epochs > 0
        assert tokens_per_epoch > 1

        # Max sequence length including any extra tokens
        max_seq_length = seq_length + add_extra_token_to_sequence

        sample_idx_entries = []

        # Loop over epochs
        for epoch in range(num_epochs):
            # Loop over documents
            for doc_idx_index in range(len(doc_idx)):
                doc_id = doc_idx[doc_idx_index]
                doc_length = sizes[doc_id]
                doc_offset = 0

                # Skip empty documents
                if doc_length == 0:
                    continue

                # While there is more document to process
                while doc_offset < doc_length:
                    remaining_length = doc_length - doc_offset
                    sample_length = min(max_seq_length, remaining_length)

                    if drop_last_partial_sequence and sample_length < max_seq_length:
                        # Skip this sample if partial sequences are to be dropped
                        break  # Move to the next document

                    # Record the starting point of the sample
                    sample_idx_entries.append([doc_idx_index, doc_offset])

                    # Move to the next sample offset
                    doc_offset += sample_length

        # Convert the list to a numpy array
        if len(sample_idx_entries) > 0:
            sample_idx_entries = np.array(sample_idx_entries, dtype=np.int32)
        else:
            # Handle the case where no samples are generated
            sample_idx_entries = np.empty((0, 2), dtype=np.int32)

        num_samples = len(sample_idx_entries)

        # Build the final sample_idx array with shape [num_samples + 1, 2]
        sample_idx = np.zeros((num_samples + 1, 2), dtype=np.int32)

        # Copy the entries
        sample_idx[:-1] = sample_idx_entries

        # For the last entry, set to the end of the last document
        if num_samples > 0:
            last_doc_idx_index = sample_idx_entries[-1][0]
            last_doc_offset = sample_idx_entries[-1][1]
            last_doc_id = doc_idx[last_doc_idx_index]
            last_doc_length = sizes[last_doc_id]
            remaining_length = last_doc_length - last_doc_offset
            sample_length = min(max_seq_length, remaining_length)
            sample_idx[-1] = [last_doc_idx_index, last_doc_offset + sample_length]
        else:
            # If no samples, set the last entry to zeros
            sample_idx[-1] = [0, 0]

        return sample_idx
        # sample_index = numpy.array(sample_index, dtype=numpy.int64)
        
        # return sample_index


    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index
        
        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the shuffle index
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            get_path_to = lambda suffix: os.path.join(
                path_to_cache,
                f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}-{suffix}",
            )
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [
                        path_to_description,
                        path_to_document_index,
                        path_to_sample_index,
                        path_to_shuffle_index,
                    ],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            
            print (f'Inside gpt_dataset.py _build_document_sample_shuffle_indices: ')
            print (f'if not path_to_cache or ( ')
            # print (f'original self.num_samples: {self.num_samples}')
            # print (f'setting self.num_samples: {self.num_samples}')
            # self.num_samples = 89
            
            print (f'self.num_samples: {self.num_samples}')

            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )
            t_beg = time.time()

            sequence_length = self.config.sequence_length
            num_tokens_per_epoch = self._get_num_tokens_per_epoch()
            num_epochs = self._get_num_epochs(num_tokens_per_epoch)
            
            print (f'num_tokens_per_epoch: {num_tokens_per_epoch}')
            

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch
                    - self.config.add_extra_token_to_sequence
                ) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (
                    num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // sequence_length
                
                print (f'self.indices: {self.indices}')
                print (f'num_samples_per_epoch: {num_samples_per_epoch}')
                print (f'num_samples_from_final_epoch: {num_samples_from_final_epoch}')
                

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(
                    threshold * num_samples_per_epoch
                )

                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                )
                log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
                log_single_rank(
                    logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
                )

            log_single_rank(
                logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
            )

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the document index
            document_index = _build_document_index(
                self.indices, num_epochs, numpy_random_state, separate_final_epoch
            )

            drop_last_partial_sequence = True
            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence

            # Build the sample index
            from megatron.core.datasets import helpers

            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
            else:
                drop_last_partial_sequence = True

            assert document_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32
            if len(document_index) * 2 > len(self.dataset.sequence_lengths):
                # Heuristic: if "access density" of sequence_lengths is relatively high,
                # force loading the mmap-ed array into memory by taking a copy.
                # System performance benefits come from two aspects:
                # 1. **sequentially** pre-loading the whole file if we're gonna read a large fraction anyways.
                # 2. GIL is held when calling into c++ code; making the c++ func faster improves parallelism.
                sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
            else:
                sequence_lengths_for_cpp = self.dataset.sequence_lengths
            
            # Zixian: Oct 7 to try out query 1 document for 1 sequence. 
            # sample_index = helpers.build_sample_idx(
            #     sequence_lengths_for_cpp,
            #     document_index,
            #     sequence_length,
            #     num_epochs,
            #     num_tokens_per_epoch,
            #     drop_last_partial_sequence,
            #     self.config.add_extra_token_to_sequence,
            # )
            
            # Always allow partial sequences to ensure we get samples from all documents
            drop_last_partial_sequence = False
            sample_index = self._custom_build_sample_idx (
                self.dataset.sequence_lengths,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last_partial_sequence,
                add_extra_token_to_sequence=self.config.add_extra_token_to_sequence,
            )
            
            # Ensure we have the correct number of samples
            num_samples = sample_index.shape[0] - 1  # Subtract 1 because of the extra entry
            print (f' \n\n num_samples = {num_samples}')
            shuffle_index = _build_shuffle_index(
                num_samples=num_samples,
                total_size=num_samples,
                numpy_random_state=numpy_random_state
            )
            # Zixian: Oct 7: Ensure to incorporate new shuffle index that has new indices
            # Build the shuffle index
            # if separate_final_epoch:
            #     shuffle_index = _build_shuffle_index(
            #         num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
            #     )
            # else:
            #     shuffle_index = _build_shuffle_index(
            #         sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
            #     )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save the {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
            )
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")
            
            print (f'document_index: {document_index}')
            # for i in document_index:
            #     print (i)
            print (f'sample_index: {sample_index}')
            # for i in sample_index: 
            #     print (i)
            print (f'shuffle_index: {shuffle_index}')
            # for i in shuffle_index: 
            #     print (i)

            return document_index, sample_index, shuffle_index

        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
        )
        print (f'Inside gpt_dataset.py _build_document_sample_shuffle_indices: ')
        print (f'else ')
            
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )

        return document_index, sample_index, shuffle_index

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        # return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))
        # Zixian: Oct 8: DEBUG
        # Change the tokens per epoch accordingly to the seq_len
        return int (len (self.indices) * self.config.sequence_length)
        

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is None:
            return num_epochs
        else:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index
    """
    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle
    
    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index
    """
    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    # Zixian: Oct 7: DEBUG 
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


class MockGPTLowLevelDataset:

    seed: int = 0
    size: int = 100000
    max_sequence_length: int = 4096

    def __init__(self, tokenizer: MegatronTokenizer) -> None:
        self.tokenizer = tokenizer
        rng = numpy.random.default_rng(seed=self.seed)
        self.sequence_lengths = rng.integers(
            low=1, high=self.max_sequence_length, size=self.size, dtype=numpy.int32
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> numpy.number:
        length = self.sequence_lengths[idx]
        sample = numpy.int64(
            numpy.concatenate([numpy.arange(length - 1) + 1, [self.tokenizer.eod]])
        )
        return sample

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        if length is None:
            length = self.sequence_lengths[idx] - offset
        return self[idx][offset : offset + length]


class MockGPTDataset(GPTDataset):
    """The mock GPT dataset

    Args:
        indexed_dataset (MockGPTLowLevelDataset): The MockGPTLowLevelDataset around which to build the MockGPTDataset

        dataset_path (Optional[str]): This argument is of no consequence for the MockGPTDataset

        indices (numpy.ndarray): The set of the dataset indices to expose

        num_samples (int): The number of samples to draw from the dataset

        index_split (Split): The indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: MockGPTLowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        assert config.mock

        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: MockGPTLowLevelDataset) -> int:
        """Abstract method implementation

        Args:
            low_level_dataset (MockGPTLowLevelDataset): The underlying MockGPTLowLevelDataset

        Returns:
            int: The number of unique elements in the underlying MockGPTLowLevelDataset
        """
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: Optional[str], config: GPTDatasetConfig
    ) -> MockGPTLowLevelDataset:
        """Abstract method implementation

        Args:
            dataset_path (Optional[str]): This argument is of no consequence for the MockGPTLowLevelDataset

            config (GPTDatasetConfig): The config

        Returns:
            MockGPTLowLevelDataset: The underlying MockGPTLowLevelDataset
        """
        return MockGPTLowLevelDataset(config.tokenizer)
