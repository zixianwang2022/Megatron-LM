# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/

import math
import torch

from dataclasses import dataclass
from functools import partial
from torch import nn, Tensor
from typing import Union

from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

from megatron.core import parallel_state
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.ssm.mamba_hybrid_layer_allocation import (
    allocate_layers, Symbols as LayerSymbols
)

from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel import get_cuda_rng_tracker


def create_mamba_block(
    config,
    mamba_layer_spec,
    ssm_cfg=None,
    residual_in_fp32=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    block = build_module(
        mamba_layer_spec,
        config,
        residual_in_fp32=residual_in_fp32,
        layer_idx=layer_idx,
        **ssm_cfg,
        **factory_kwargs
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    with get_cuda_rng_tracker().fork():
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        for name, p in module.named_parameters():
            if name in ["in_proj.weight", "x_proj.weight", "conv1d.weight", "out_proj.weight"]:
                nn.init.kaiming_uniform(p, a=math.sqrt(5))

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


@dataclass
class MambaStackSubmodules:
    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp


class MambaStack(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaStackSubmodules,
        rms_norm: bool = False,
        initializer_cfg=None,
        residual_in_fp32=False,
        pre_process: bool = True,
        hybrid_attention_ratio: float = 0.0,
        hybrid_mlp_ratio: float = 0.0,
        hybrid_override_pattern: str = None,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config)
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        # TODO (duncan): potentially move hybrid_attention_ratio,
        #   hybrid_mlp_ratio, and hybrid_override_pattern into config
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.hybrid_override_pattern = hybrid_override_pattern

        layer_type_list = allocate_layers(
            self.config.num_layers, self.hybrid_attention_ratio,
            self.hybrid_mlp_ratio, self.hybrid_override_pattern)

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            raise ValueError("Pipeline parallel not currently implemented for "
                             "Mamba1 MambaStack")

        self.layers = nn.ModuleList()
        for i in range(self.config.num_layers):
            layer_type = layer_type_list[i]
            if layer_type == LayerSymbols.MAMBA:
                block = create_mamba_block(
                    self.config,
                    submodules.mamba_layer,
                    residual_in_fp32=residual_in_fp32,
                    layer_idx=i,
                    **factory_kwargs,
                )
            elif layer_type == LayerSymbols.ATTENTION:
                # Wondering if layer_number should be i+1. See TransformerBlock
                block = build_module(submodules.attention_layer,
                                     config=self.config, layer_number=i)
            elif layer_type == LayerSymbols.MLP:
                # Wondering if layer_number should be i+1. See TransformerBlock
                block = build_module(submodules.mlp_layer, config=self.config,
                                     layer_number=i)
            else:
                assert True, "unexpected layer_type"
            self.layers.append(block)

        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

        self.final_norm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=self.config.num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_params=None,
        rotary_pos_emb: Tensor=None,
        **kwargs
    ):
        if hidden_states == None:
            hidden_states = self.input_tensor

        if inference_params:
            # NOTE(bnorick): match InferenceParams attributes for mamba_ssm.utils.generation.InferenceParams,
            # this hack supports eval
            inference_params.max_seqlen = inference_params.max_sequence_length
            inference_params.seqlen_offset = inference_params.sequence_len_offset

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
            )

            # The attention layer (currently a simplified transformer layer)
            # outputs a tuple of (hidden_states, context). Context is intended
            # for cross-attention, and is not needed in our model.
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        hidden_states = self.final_norm(hidden_states)

        return hidden_states
