import math
import torch
from functools import partial
from torch import nn, Tensor
from megatron import print_rank_0
from megatron.core.transformer.module import MegatronModule
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.ssm.mamba_layer import MambaLayer


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
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

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

def allocate_attention(total_layers_count: int, attention_ratio: float):
    assert total_layers_count > 0
    assert attention_ratio >= 0.0 and attention_ratio <= 1.0

    attention_layers_count: int = round(total_layers_count * attention_ratio)
    mamba_layers_count: int = total_layers_count - attention_layers_count
    mamba_sections_count: int = attention_layers_count + 1
    mamba_section_length: float = mamba_layers_count / mamba_sections_count

    layer_is_attention = [False] * total_layers_count
    i: int = mamba_section_length
    for l in range(total_layers_count):
        if i < 0.5:
            layer_is_attention[l] = True
            i += mamba_section_length
        else:
            i -= 1

    if attention_ratio > 0.0:
        actual_attention_layers_count = sum(layer_is_attention)
        actual_attention_ratio = (actual_attention_layers_count /
                                  total_layers_count)
        allocation = ''.join(['*' if a else 'M' for a in layer_is_attention])
        print_rank_0("Hybrid allocation (* represents an attention layer):")
        print_rank_0(allocation)
        print_rank_0(f"{actual_attention_layers_count} attention layers in "
                     f"{total_layers_count} total layers. Actual attention "
                     f"ratio: {actual_attention_ratio:.2f}")

    return layer_is_attention


class MambaStack(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        mamba_layer_spec: ModuleSpec,
        attention_layer_spec: ModuleSpec,
        rms_norm: bool = False,
        initializer_cfg=None,
        residual_in_fp32=False,
        hybrid_attention_ratio: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config)
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        # TODO (duncan): potentially move percent_attention into config
        self.hybrid_attention_ratio = hybrid_attention_ratio

        layer_is_attention = allocate_attention(
            self.config.num_layers, self.hybrid_attention_ratio)

        self.layers = nn.ModuleList()
        for i in range(self.config.num_layers):
            if layer_is_attention[i]:
                # Wondering if layer_number should be i+1. See TransformerBlock
                block = build_module(attention_layer_spec, config=self.config,
                                     layer_number=i)
            else:
                block = create_mamba_block(
                    self.config,
                    mamba_layer_spec,
                    residual_in_fp32=residual_in_fp32,
                    layer_idx=i,
                    **factory_kwargs,
                )
            self.layers.append(block)

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
            # Option 2 to change the hidden states tensor format (and only
            # option if using non-TE attention)
            # `TransformerLayer` expects the inputs in [s, b, d] format
            # if isinstance(layer, TransformerLayer):
            #     hidden_states = hidden_states.transpose(0,1).contiguous()

            hidden_states = layer(
                hidden_states, attention_mask, inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
            )

            # The attention layer (currently a simplified transformer layer)
            # outputs a tuple of (hidden_states, context). Context is intended
            # for cross-attention, and is not needed in our model.
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            # Option 2 to change the hidden states tensor format(and only
            # option if using non-TE attention)
            # `TransformerLayer` outputs in [s, b, d] format. Convert back to
            # `[b, s, d]` format
            # if isinstance(layer, TransformerLayer):
            #     hidden_states = hidden_states.transpose(0,1).contiguous()

        hidden_states = self.final_norm(hidden_states)

        return hidden_states
