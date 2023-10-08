# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""

import torch

from megatron import get_args
from megatron.core import tensor_parallel
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model

import math
from tools.mup.shape import set_base_shapes
from tools.mup.init import normal_
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):
    args = get_args()
    output_temperature = args.output_temperature

    if hasattr(logit_weights, 'infshape'):
        width_mult = logit_weights.infshape.width_mult()
    else:
        width_mult = 1.0

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output / width_mult * output_temperature,
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
    
"""
The only things we modified from the original GPTModel are 
1) replace the readout layer with MuReadout or MuSharedReadout, --- > 
 the initialization of self.output_layer is ok, only change the \
post_language_model_processing() function
2) use fan_in style initialization, 
3) change attention scaling to 1/d instead of 1/sqrt(d), and
4) zero initialization of query weights 
5) ?? and also zero initialization of the output layer?? (in appendix)
6) muT style optimizer and scheduler
7) add multiplier to the parameterization initialization (
    a) initialization multiplier, 
    b) embedding multiplier, 
    c) attention temperature, 
    d) output temperature, 
    (not this one) relative position embedding multiplier)
"""


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

        if args.muT_config_file is None:
            self._fanin_initialization(args)

    def _fanin_initialization(self, config):
        if hasattr(config, "shape_file"):
            print_rank_0("Set the base shapes!")

            set_base_shapes(self, config.shape_file, rescale_params=False)

            # here manually initialize all the named parameters with the muTranfer normal initializer
            for name, tensor in self.named_parameters():
                print_rank_0("The name is {}".format(name))
                if name.endswith('.dense_4h_to_h.weight') or name.endswith('.dense.weight'):
                    # initialize all the output dense matrix weight
                    print_rank_0("reset the initialization for output dense matrix!")
                    std = config.init_method_std / math.sqrt(2.0 * config.num_layers)
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        normal_(tensor, 0 , std)
                elif name.endswith('layernorm.weight'):
                    # initialize all the layer norm weight as all 1, and bias as 0
                    if tensor.std() != 0 and tensor.mean() != 1:
                        raise ValueError(f'need to check {name} init')
                    print_rank_0("reset the initialization for layernorm weight!")
                    if config.apply_layernorm_1p:
                        normal_(tensor, 0, 0)
                    else:
                        normal_(tensor, 1, 0)
                elif name.endswith('layernorm.bias'):
                    if tensor.std() !=0 and tensor.mean() !=0:
                        raise ValueError(f'need to check {name} init')
                    print_rank_0("reset the initialization for layernorm bias!")
                    normal_(tensor, 0, 0)
                elif name.endswith('.weight'):
                    # initialize all the other dense matrix weight
                    print_rank_0("reset .weight with muP style initialization!")
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        normal_(tensor, 0, config.init_method_std)
                else:
                    if tensor.std() != 0 and tensor.mean() != 0:
                        raise ValueError(f'need to check {name} init')
                    
                # initialization scale
                init_scale = config.initialization_scale
                print_rank_0("(1) Multiply the parameter with init_scale {}".format(init_scale))
                tensor.data *= init_scale

        return


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
