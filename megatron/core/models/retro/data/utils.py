# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
import torch


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_config_path(workdir):
    '''Argument copy stored within retro workdir.'''
    return os.path.join(workdir, "config.json")


def get_num_chunks_per_sample(config):
    '''Compute seq_length // chunk_length.'''
    sample_length = config.retro_gpt_seq_length
    chunk_length = config.retro_gpt_chunk_length
    assert sample_length % chunk_length == 0
    return sample_length // chunk_length


class GPTToTextDataset(torch.utils.data.Dataset):
    '''Dataset to convert GPT tokens to text.'''

    def __init__(self, gpt_dataset):

        super().__init__()

        self.gpt_dataset = gpt_dataset
        self.gpt_tokenizer = get_gpt_tokenizer()

    def __len__(self):
        return len(self.gpt_dataset)

    def __getitem__(self, idx):
        gpt_token_ids = self.gpt_dataset[idx]["text"].tolist()
        text = self.gpt_tokenizer.detokenize(gpt_token_ids)
        return {"text": text}
