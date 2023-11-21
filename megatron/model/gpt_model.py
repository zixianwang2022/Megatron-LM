# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""

import math
import torch

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model

from tools.mup.shape import set_base_shapes
from tools.mup.init import normal_ as mup_normal_


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):
    args = get_args()
    if args.use_mup:
        assert hasattr(logit_weights, 'infshape'), \
            'The infshape for logit_weights is missing!'
        width_mult = logit_weights.infshape.width_mult()
        lm_output = lm_output / width_mult
    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

        if args.use_mup:
            self.mup_rescale_initializtaions(args)

    def mup_rescale_initializtaions(self, args):
        """Set initializations according to muP (Table 8)."""
        assert hasattr(args, 'shape_file'), \
            'muP is enabled but a shape file is missing.'

        set_base_shapes(self, args.shape_file)

        for name, tensor in self.named_parameters():
            if name.endswith('.dense_4h_to_h.weight') or name.endswith('.dense.weight'):
                # Set the initialization scales for the output layer of each block.
                if args.strict_fan_in_init:
                    # Optionally remove depth-wise scaling for initialization,
                    # to be consistent with muP.
                    std = args.init_method_std
                else:
                    # Default of Megatron.
                    print("Warning: using depth-wise initialization for block output layers.")
                    std = args.init_method_std / math.sqrt(2.0 * args.num_layers)
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    if name.endswith('.dense_4h_to_h.weight'):
                        if args.strict_fan_in_init:
                            # Need to divide the global std by a factor of sqrt(in_dim/out_dim),
                            # because the output dim of FFN is consistent across Transformer layers,
                            # but the input dim can change for swiglu vs. other activations.
                            out_dim = tensor.infshape[0].dim
                            in_dim = tensor.infshape[1].dim
                            std_div = math.sqrt(in_dim / out_dim)
                        else:
                            std_div = 1.
                        mup_normal_(tensor, 0, std / std_div)
                    else:
                        mup_normal_(tensor, 0 , std)
            elif name.endswith('layernorm.weight'):
                # Effectively initialize all the layer norm weights to 1.
                if args.apply_layernorm_1p:
                    torch.init.zero_(tensor)
                else:
                    torch.init.ones_(tensor)
            elif name.endswith('layernorm.bias'):
                torch.init.zero_(tensor)
            elif name.endswith('.weight'):
                # Apply width-dependent initialization to matrice-like weights.
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    mup_normal_(tensor, 0, args.init_method_std)
            else:
                assert torch.all(tensor == 0), \
                    f'Found non-zero init for {tensor.var_name}, which is supposed to be vector_like (shape: {tensor.shape}).'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None):

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            return post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
