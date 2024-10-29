# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional

from torch import Tensor

import os
from megatron.training import get_args
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


class MambaModel(LanguageModule):
    """Mamba language model.

    Args:
        config (TransformerConfig): Transformer config
        mamba_stack_spec (ModuleSpec): Specifies the modules to use for the various layer types
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        mamba_ssm_ngroups (int, optional): Specifies the number of groups to use. The default value is 8, as in the NVIDIA Mamba2 (pure and hybrid) 8b. However, in the original Mamba2 paper, the checkpoints use a setting of 1. Defaults to 8.
        hybrid_attention_ratio (float, optional): The target ratio of attention layers to total layers
        hybrid_mlp_ratio (float, optional): The target ratio of mlp layers to total layers
        hybrid_override_pattern (str, optional): The hybrid layer pattern to override with
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope,none], optional):  Position embedding type. Defaults to 'none'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        mamba_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        mamba_ssm_ngroups: int = 8,
        pre_process: bool = True,
        hybrid_attention_ratio: float = 0.0,
        hybrid_mlp_ratio: float = 0.0,
        hybrid_override_pattern: str = None,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        # Mamba with no attention has no need for position embeddings, so none is default
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.mamba_stack_spec: ModuleSpec = mamba_stack_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.mamba_ssm_ngroups = mamba_ssm_ngroups
        self.pre_process = pre_process
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.hybrid_override_pattern = hybrid_override_pattern
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        self.decoder = build_module(
            mamba_stack_spec,
            self.config,
            mamba_ssm_ngroups=self.mamba_ssm_ngroups,
            pre_process=self.pre_process,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            post_process=self.post_process,
            dtype=config.params_dtype,
        )

        # Output
        if post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        
        # For retrieving states
        insert_states: bool =False, 
        retrieve_states: bool =False, 
        insert_states_for_training: bool = False, 
        # inserted_all_states: Tensor =None, 
    ) -> Tensor:
        """Forward function of the Mamba model. This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units
        """
        
        soup_train = os.getenv ('SOUP_TRAIN')
        
        print (f'[mamba_model.py]: input_ids.shape: {input_ids.shape}')
    
        # args = get_args()
        # insert_states = args.inserting_mamba_states
        # retrieve_states = args.retrieving_mamba_states
        # insert_states_for_training = args.insert_mamba_states_for_training

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # The following assert will currently fail when running inference.
        # Commented out for now.
        # TODO (duncan/rwaleffe): (1) confirm that the externally-generated
        #   attention mask is not needed and is ignored by the model in
        #   inference mode, (2) reduce the size of the externally-generated
        #   attention mask to prevent CPU OOM (as we did for training), (3)
        #   force the attention mask passed to the model in inference mode to
        #   be None, so this assert will succeed.
        # assert attention_mask is None, "The attention mask is ignored and should be set to None"


        if soup_train:
            insert_states = False 
            retrieve_states = True 
            insert_states_for_training = True 
            inserted_all_states = None 
            
        
        
        # Run decoder.
        # Extract States 
        print (f'[mamba_model.py]: Entering FIRST forward-pass')
        hidden_states, all_layers_states_dict = self.decoder(
                                                        hidden_states=decoder_input,
                                                        attention_mask=attention_mask,
                                                        inference_params=inference_params,
                                                        rotary_pos_emb=rotary_pos_emb,
                                                        
                                                        insert_states=insert_states, 
                                                        retrieve_states=retrieve_states, 
                                                        inserted_all_states=inserted_all_states, 
                                                        insert_states_for_training=insert_states_for_training, 
                                                        # Zixian: Oct 28: Not used in new version of Megatron Mamba
                                                        # **(extra_block_kwargs or {}),
                                                        )
        # TODO: include states for layers besides layer 1
        print (f'[mamba_model.py]: all_layers_states_dict[0][0].keys(): {all_layers_states_dict[0][0].keys()}')
        print (f'[mamba_model.py]: all_layers_states_dict[0][0]["ssm_state"].shape: {all_layers_states_dict[0][0]["ssm_state"].shape}')
        print (f'[mamba_model.py]: all_layers_states_dict[0][0]["ssm_state"][:][0][0]: {all_layers_states_dict[0][0]["ssm_state"][:][0][0]}')
        
        
        if soup_train: 
            # Soup states 
            # TODO: 
        
            insert_states = True 
            retrieve_states = False  
            insert_states_for_training = True 
            inserted_all_states = all_layers_states_dict 
            
            print (f'[mamba_model.py]: Entering SECOND forward-pass')
            hidden_states, all_layers_states_dict = self.decoder(
                                                        hidden_states=decoder_input,
                                                        attention_mask=attention_mask,
                                                        inference_params=inference_params,
                                                        rotary_pos_emb=rotary_pos_emb,
                                                        
                                                        insert_states=insert_states, 
                                                        retrieve_states=retrieve_states, 
                                                        inserted_all_states=inserted_all_states, 
                                                        insert_states_for_training=insert_states_for_training, 
                                                        # Zixian: Oct 28: Not used in new version of Megatron Mamba
                                                        # **(extra_block_kwargs or {}),
                                                        )
            
        

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss
