# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from mamba_ssm import Mamba

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

mamba_layer_spec = ModuleSpec(
        module=MambaLayer,
        submodules=MambaLayerSubmodules(
            norm=TENorm,
            mixer=Mamba
        ),
)

# Started with spec from gpt_layer_specs.py (with MLP removed)
# TODO (duncan/sudhakar): make it so only one spec must be specified/loaded
attention_layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
        ),
)
