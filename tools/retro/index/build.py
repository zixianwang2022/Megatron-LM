# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from .add import add_to_index
from .train import train_index


def build_index():
    '''Train & add to index.'''
    train_index()
    add_to_index()
