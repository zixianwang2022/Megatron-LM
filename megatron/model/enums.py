# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import enum

class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2
    # >>>
    retro_encoder = 3
    retro_decoder = 4
    # <<<

class LayerType(enum.Enum):
    encoder = 1
    decoder = 2
    # >>>
    # retro_encoder_ca = 3
    # retro_decoder_sa = 4
    # retro_decoder_ca = 5
    retro_encoder = 3
    retro_decoder_first = 4
    retro_decoder_other = 5
    # <<<
 
class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2

class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
