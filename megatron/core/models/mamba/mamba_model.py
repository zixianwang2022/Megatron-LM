# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional

from torch import Tensor

import os
import torch
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
        
        self.all_layers_states_dict = {}
        
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
            
            
        # Zixian: Nov 2: Initialize total gradient norm accumulator
        self.total_grad_norm = 0.0
        # Register hooks on all parameters
        # self._register_param_hooks()
            

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
        
        
        
    def soup_states_origin (self, states_dict): 
        """
        soup_states

        Args:
            states_dict (Dict):     {1: {'conv_state': [batch, ...]}
                                        {'ssm_state' : [batch, ...]}
                                     2: {'conv_state': [batch, ...]}
                                        {'ssm_state' : [batch, ...]}
                                     3: ...}

        Returns:
            states_dict (Dict):     {1: {'conv_state': [1, ...]}
                                        {'ssm_state' : [1, ...]}
                                     2: {'conv_state': [1, ...]}
                                        {'ssm_state' : [1, ...]}
                                     3: ...}
        """
        
        output_state_whole = {}
        
        for layer_id, states in states_dict.items():
            summed_states = {}
            
            # Soup by average 
            soupped_conv = torch.mean(states['conv_state'], dim=0, keepdim=True)  # Shape: [1, ...]
            soupped_ssm = torch.mean(states['ssm_state'], dim=0, keepdim=True)  # Shape: [1, ...]
            
            
            # # Zixian: Oct 30: Debug to test if states are soupped correctly
            # if (layer_id == 4): 
            #     # Zixian: Oct 30: Verified that soupped states are the sames
                
            #     print (f'DEBUG: mamba_model.py @ soup_states] states["ssm_state"].shape: \n{states["ssm_state"].shape}')
            #     print (f'DEBUG: mamba_model.py @ soup_states] soupped_ssm.shape: \n{soupped_ssm.shape}')
            #     # print (f'DEBUG: mamba_model.py @ soup_states] states["ssm_state"][1]: \n{states["ssm_state"][1]}')
            #     print (f'DEBUG: mamba_model.py @ soup_states] states["ssm_state"][0] + states["ssm_state"][1]: \n{(states["ssm_state"][0] + states["ssm_state"][1]) / 2}')
            #     print (f'DEBUG: mamba_model.py @ soup_states] soupped_conv[0]: \n{soupped_ssm[0]}')
                
            #     print (f'DEBUG: mamba_model.py @ soup_states] states["conv_state"].shape: \n{states["conv_state"].shape}')
            #     print (f'DEBUG: mamba_model.py @ soup_states] soupped_conv.shape: \n{soupped_conv.shape}')
            #     # print (f'DEBUG: mamba_model.py @ soup_states] states["conv_state"][1]: \n{states["conv_state"][1]}')
            #     print (f'DEBUG: mamba_model.py @ soup_states] states["conv_state"][0] + states["conv_state"][1]: \n{(states["conv_state"][0] + states["conv_state"][1]) /2}')
            #     print (f'DEBUG: mamba_model.py @ soup_states] soupped_conv[0]: \n{soupped_conv[0]}')
                
                
            
            # Store to a new dict 
            summed_states['conv_state'] = soupped_conv
            summed_states['ssm_state'] = soupped_ssm
            
            # Store for each layer 
            output_state_whole[layer_id] = summed_states
            
        return output_state_whole
    
    
    
    
    def split_doc_batch(self, input_ids_batch: Tensor) -> Tensor:
        """
        Splits the input_ids tensor into document chunks based on a specific pattern.
        Each chunk is padded to seqlen with the padding token `3`.
        
        Parameters:
        - input_ids_batch (torch.Tensor): A tensor of shape (1, seqlen) on a specific device.
        
        Returns:
        - document_batch (torch.Tensor): A tensor containing the document padded chunks of shape (num_segments, seqlen).
        """
        # Define the pattern to split on
        pattern = torch.tensor([44354, 251594, 226308, 251621], device=input_ids_batch.device)
        pattern_length = pattern.size(0)
        seqlen = input_ids_batch.size(1)
        padding_token = 3

        # Create sliding windows of size equal to the pattern length
        windows = input_ids_batch.unfold(1, pattern_length, 1)  # Shape: [1, seqlen - pattern_length + 1, pattern_length]

        # Check where the pattern matches
        matches = (windows == pattern).all(dim=2)  # Shape: [1, seqlen - pattern_length + 1]

        # Find the starting indices of the pattern
        match_indices = torch.nonzero(matches, as_tuple=False)[:, 1]  # Shape: [num_matches]

        # Initialize variables to store split points
        split_points = []
        previous_end = 0

        # Iterate over each match to determine split points
        for match_idx in match_indices:
            # Extract the segment before the current pattern
            segment = input_ids_batch[:, previous_end:match_idx]
            if segment.size(1) > 0:
                split_points.append(segment.squeeze(0))  # Shape: [L]
            # Update the previous_end to be after the current pattern
            previous_end = match_idx + pattern_length

        # After processing all matches, check if there's a segment after the last pattern
        if previous_end < seqlen:
            segment = input_ids_batch[:, previous_end:]
            if segment.size(1) > 0:
                split_points.append(segment.squeeze(0))  # Shape: [L]

        # If no patterns were found, the entire input is one segment
        if not split_points:
            split_points = [input_ids_batch.squeeze(0)]  # Shape: [seqlen]

        # Pad each segment to the desired seqlen
        padded_segments = []
        for segment in split_points:
            pad_length = seqlen - segment.size(0)
            if pad_length > 0:
                pad = torch.full((pad_length,), padding_token, device=input_ids_batch.device)
                padded = torch.cat((segment, pad), dim=0)  # Shape: [seqlen]
            else:
                padded = segment[:seqlen]  # Truncate if necessary
            padded_segments.append(padded)

        # Stack all padded segments into a single batch tensor
        if padded_segments:
            document_batch = torch.stack(padded_segments, dim=0)  # Shape: [num_segments, seqlen]
        else:
            # If no segments are present, return an empty tensor
            document_batch = torch.empty((0, seqlen), device=input_ids_batch.device)

            
        # Debug
        for i in range (document_batch.shape[0]):
            if i%matches == 0: 
                print (f"[mamba_model.py split_doc_batch] input_ids_batch[0][:300]: {i%matches[i%matches][:300]}")
            print (f"[mamba_model.py split_doc_batch] document_batch[{i}][:100]: {document_batch[i][:100]}")
        
        print (f"[mamba_model.py split_doc_batch] document_batch.shape: {document_batch.shape}") 
        
        return document_batch
    
    
    
    
    
    def get_doc_qa_pair (self, input_ids_batch: Tensor) :
        """
        Splits the input_ids tensor into document chunks based on a specific pattern for each batch.
        Each chunk is padded to seqlen with the padding token `3`.

        Parameters:
        - input_ids_batch (torch.Tensor): A tensor of shape (batch_size, seqlen) on a specific device.

        Returns:
        - first_chunks (torch.Tensor): A tensor containing all but the last QA chunks for each batch.
                                    Shape: [batch_size * (num_patterns), seqlen]
        - last_chunks (torch.Tensor): A tensor containing the last QA chunk for each batch.
                                    Shape: [batch_size, seqlen]
        """
        batch_size, seqlen = input_ids_batch.size()
        padding_token = 3

        # Define the pattern to split on
        pattern = torch.tensor([44354, 251594, 226308, 251621], device=input_ids_batch.device)
        pattern_length = pattern.size(0)

        # Initialize lists to store all first chunks and last chunks
        all_first_chunks = []
        all_last_chunks = []

        for batch_idx in range(batch_size):
            input_ids = input_ids_batch[batch_idx].unsqueeze(0)  # Shape: [1, seqlen]

            # Create sliding windows of size equal to the pattern length
            windows = input_ids.unfold(1, pattern_length, 1)  # Shape: [1, seqlen - pattern_length + 1, pattern_length]

            # Check where the pattern matches
            matches = (windows == pattern).all(dim=2)  # Shape: [1, seqlen - pattern_length + 1]

            # Find the starting indices of the pattern
            match_indices = torch.nonzero(matches, as_tuple=False)[:, 1]  # Shape: [num_matches]

            # Initialize variables to store split points
            split_points = []
            previous_end = 0

            # Iterate over each match to determine split points
            for match_idx in match_indices:
                # Extract the segment before the current pattern
                segment = input_ids[:, previous_end:match_idx]
                if segment.size(1) > 0:
                    split_points.append(segment.squeeze(0))  # Shape: [L]
                # Update the previous_end to be after the current pattern
                previous_end = match_idx + pattern_length

            # After processing all matches, check if there's a segment after the last pattern
            if previous_end < seqlen:
                segment = input_ids[:, previous_end:]
                if segment.size(1) > 0:
                    split_points.append(segment.squeeze(0))  # Shape: [L]

            # If no patterns were found, the entire input is one segment
            if not split_points:
                split_points = [input_ids.squeeze(0)]  # Shape: [seqlen]

            # Separate all but the last chunk and the last chunk
            first_chunks = split_points[:-1]
            last_chunk = split_points[-1]

            # Pad first_chunks
            for segment in first_chunks:
                pad_length = seqlen - segment.size(0)
                if pad_length > 0:
                    pad = torch.full((pad_length,), padding_token, device=input_ids_batch.device)
                    padded = torch.cat((segment, pad), dim=0)  # Shape: [seqlen]
                else:
                    padded = segment[:seqlen]  # Truncate if necessary
                all_first_chunks.append(padded)

            # Pad last_chunk
            pad_length = seqlen - last_chunk.size(0)
            if pad_length > 0:
                pad = torch.full((pad_length,), padding_token, device=input_ids_batch.device)
                padded_last = torch.cat((last_chunk, pad), dim=0)  # Shape: [seqlen]
            else:
                padded_last = last_chunk[:seqlen]  # Truncate if necessary
            all_last_chunks.append(padded_last)

        # Stack all first_chunks and last_chunks into tensors
        if all_first_chunks:
            first_chunks_tensor = torch.stack(all_first_chunks, dim=0)  # Shape: [batch_size * (num_patterns), seqlen]
        else:
            first_chunks_tensor = torch.empty((0, seqlen), device=input_ids_batch.device)

        if all_last_chunks:
            last_chunks_tensor = torch.stack(all_last_chunks, dim=0)  # Shape: [batch_size, seqlen]
        else:
            last_chunks_tensor = torch.empty((0, seqlen), device=input_ids_batch.device)

        # Debug
        # Zixian Nov 5: Verified 
        # if input_ids_batch.device == torch.device ("cuda:0"): 
        #     print (f"[mamba_model.py split_doc_batch] first_chunks_tensor.shape: {first_chunks_tensor.shape}") 
        #     print (f"[mamba_model.py split_doc_batch] last_chunks_tensor.shape: {last_chunks_tensor.shape}") 
        #     for i in range (first_chunks_tensor.shape[0]):
        #         # if i<batch_size and (batch_size != 1): 
        #         print (f"[mamba_model.py split_doc_batch] input_ids_batch[{int(i//batch_size)}][:400]: {input_ids_batch[int (i//batch_size)][:400]}")
        #         print (f"[mamba_model.py split_doc_batch] last_chunks_tensor[{int(i//batch_size)}][:150]: {last_chunks_tensor[int(i//batch_size)][:150]}")
        #         print (f"[mamba_model.py split_doc_batch] first_chunks_tensor[{i}][:150]: {first_chunks_tensor[i][:150]}")
            
            

        return first_chunks_tensor, last_chunks_tensor
    
    
    
    
    
    def soup_states (self, states_dict, desired_batch_size):
        """
        soup_states with grouped averaging

        Args:
            states_dict (Dict): {
                1: {'conv_state': [batch, ...], 'ssm_state': [batch, ...]},
                2: {'conv_state': [batch, ...], 'ssm_state': [batch, ...]},
                3: ...}
            desired_batch_size (int): The target batch size after averaging.
            
            assert (batch % desired_batch_size == 0, "Must group an integer number of documents' states into 1 states for all batch ")

        Returns:
            states_dict (Dict): {
                1: {'conv_state': [desired_batch_size, ...], 'ssm_state': [desired_batch_size, ...]},
                2: {'conv_state': [desired_batch_size, ...], 'ssm_state': [desired_batch_size, ...]},
                3: ...}
        """
        
        output_state_whole = {}
        
        for layer_id, states in states_dict.items():
            summed_states = {}
            
            # Get original batch size
            original_batch_size = states['conv_state'].size(0)
            
            # Ensure the original batch size is divisible by the desired batch size
            if original_batch_size % desired_batch_size != 0:
                raise ValueError(f"Original batch size ({original_batch_size}) is not divisible by desired batch size ({desired_batch_size}).")
            
            group_size = original_batch_size // desired_batch_size
            
            
            
            # if (layer_id == 5) and (states['conv_state'].device==torch.device("cuda:0")):
            #     print (f'original_batch_size: {original_batch_size}')
            #     print (f'desired_batch_size: {desired_batch_size}')
            #     print (f'group_size: {group_size}')
                
            #     print(f'DEBUG: layer {layer_id}')
            #     print(f'states["conv_state"].shape: {states["conv_state"].shape}')
            #     print(f'states["ssm_state"].shape: {states["ssm_state"].shape}')
            #     print(f'states["conv_state"][0][0]: {states["conv_state"][0]}')
            #     print(f'states["ssm_state"][0][0]: {states["ssm_state"][0]}')
            #     print(f'states["conv_state"][1][0]: {states["conv_state"][1]}')
            #     print(f'states["ssm_state"][1][0]: {states["ssm_state"][1]}')
            #     print(f'states["conv_state"][2][0]: {states["conv_state"][2]}')
            #     print(f'states["ssm_state"][2][0]: {states["ssm_state"][2]}')
            #     print(f'states["conv_state"][3][0]: {states["conv_state"][3]}')
            #     print(f'states["ssm_state"][3][0]: {states["ssm_state"][3]}')
                
            
            # Reshape and average
            # For conv_state
            # Original shape: [N, ...]
            # Reshaped shape: [M, group_size, ...]
            # Averaged shape: [M, ...]
            soupped_conv_a = states['conv_state'].view(desired_batch_size, group_size, *states['conv_state'].shape[1:])
            soupped_conv = torch.mean(soupped_conv_a, dim=1)  # Shape: [M, ...]

            # For ssm_state
            soupped_ssm_a = states['ssm_state'].view(desired_batch_size, group_size, *states['ssm_state'].shape[1:])
            soupped_ssm = torch.mean(soupped_ssm_a, dim=1)  # Shape: [M, ...]

            # Optional: Debugging statements (similar to the original code)
            # if (layer_id == 5) and (states['conv_state'].device==torch.device("cuda:0")):
            #     print(f'DEBUG: layer {layer_id}')
            #     print(f'soupped_conv.shape: {soupped_conv.shape}')
            #     print(f'soupped_ssm.shape: {soupped_ssm.shape}')
            #     print(f'soupped_conv[0][0]: {soupped_conv[0]}')
            #     print(f'soupped_ssm[0][0]: {soupped_ssm[0]}')
            #     print(f'soupped_conv[1][0]: {soupped_conv[1]}')
            #     print(f'soupped_ssm[1][0]: {soupped_ssm[1]}')
                

            # Store to a new dict
            summed_states['conv_state'] = soupped_conv
            summed_states['ssm_state'] = soupped_ssm
            
            # Store for each layer
            output_state_whole[layer_id] = summed_states
            
        return output_state_whole
    
    
    def _register_param_hooks(self):
        """
        Zixian: Nov2: See how much gradient is updated to params 
        
        Register all parameters' hook to calculate all param's gradient 
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(self._get_param_hook(name))
                
    def _get_param_hook(self, name):
        """
        The custom hook to register all parameters' gradient to a self.total_grad_norm

        Args:
            name (_type_): Name of the param
        """
        def hook(grad):
            grad_norm = grad.data.norm(2).item()
            self.total_grad_norm += grad_norm ** 2
            
            print(f"Registering model param hook: Gradient norm for {name}: {grad_norm}")
        return hook
    
    def _report_self_total_grad_norm (self, variable_grad, name): 
        # print (f'[mamba_model.py]: ')
        # print (f'[mamba_model.py]: Triggered when calculating gradient with respect to {name}, self.total_grad_norm on device {variable_grad.device}: {self.total_grad_norm}') 
        print (f'[mamba_model.py]: Triggered when calculating gradient with respect to {name}, on device {variable_grad.device}') 

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
        
        # If not the first iteration, print the total gradient norm from the previous backward pass
        if self.total_grad_norm != 0.0:
            total_gradient_norm = self.total_grad_norm ** 0.5
            print(f"Total parameter gradient norm from previous step on device {input_ids.device}:", total_gradient_norm)
        
        # Zixian: Nov 2: Reset the total gradient norm accumulator each step 
        self.total_grad_norm = 0.0
        
        # Reading Global Params for SOUP Training
        soup_train = os.getenv ('SOUP_TRAIN')
        soup_doc_num = int (os.getenv ('SOUP_DOC_NUM'))
        soup_doc_sep_token_id = int (os.getenv ('SOUP_DOC_SEP_TOKEN_ID')) 
        
        
        # print (f'[mamba_model.py]: input_ids.shape: {input_ids.shape}')
        print (f'[mamba_model.py]: input_ids[0][:300]: {input_ids[0][:300]}')
        
        
        
        # TODO: Split the batchsize 1 input sequence into #documents batch 
        # Zixian: Oct 30: put everything into 1 then extract the question batch after embedding 
        # document_batch, question_batch = input_ids ()
        if (soup_train): 
            documents_batch, qa_batch = self.get_doc_qa_pair (input_ids)
            
            # raise (RuntimeError, 'manual error')
        batcsize = input_ids.shape[0]
    

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            
            # Zixian: Oct 30: embed both documents_batch and qa_bath
            if (soup_train): 
                decoder_input = self.embedding(input_ids=documents_batch, position_ids=position_ids)
                qa_batch_input = self.embedding(input_ids=qa_batch, position_ids=position_ids)
            else: 
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
            
        # for name, param in self.named_parameters(): 
        #     print (f'[mamba_model.py]: for name, param in self.named_parameters(): ')
        #     print (f'[mamba_model.py]: name: {name}')
        #     print (f'[mamba_model.py]: param: {param}')
            
        # with torch.no_grad (): 
            
        # Run decoder.
        # Extract States 
        print (f'[mamba_model.py]: Entering FIRST forward-pass')
        hidden_states, self.all_layers_states_dict = self.decoder(
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
        
        print (f'[mamba_model.py]: After FIRST forward-pass')
        # try: 
        #     decoder_input.register_hook(lambda grad: self._report_self_total_grad_norm (grad, "decoder_input from FIRST forward"))
        # except:
        #     print (f'[mamba_model.py]: failed to register hook to self.all_layers_states_dict or all_soupped_states. Could be due to evaluation mode.')
        
        # print (f'[mamba_model.py]: all_layers_states_dict[1].keys(): {all_layers_states_dict[1].keys()}')
        # print (f'[mamba_model.py]: all_layers_states_dict[1]["ssm_state"].shape: {all_layers_states_dict[1]["ssm_state"].shape}')
        # print (f'[mamba_model.py]: all_layers_states_dict[1]["ssm_state"][:][0][0]: {all_layers_states_dict[1]["ssm_state"][:][0][0]}')
        
        
        if soup_train: 
            # Soup states 
            # TODO: 
            all_soupped_states = self.soup_states (self.all_layers_states_dict, desired_batch_size=batcsize)
            
            # raise (RuntimeError, 'manual error')
            # print (f'[mamba_model.py]: qa_batch_input.shape: {decoder_input.shape}')
            # print (f'[mamba_model.py]: split_docs_batched[:-1].shape: {split_docs_batched[:-1].shape}\n')
            # print (f'[mamba_model.py]: qa_batch_input.shape: {qa_batch_input.shape}')
            # print (f'[mamba_model.py]: split_docs_batched[-1:].shape: {split_docs_batched[-1:].shape}')
            
            # print (f'[mamba_model.py]: all_soupped_states[1]["ssm_state"].shape: {all_soupped_states[1]["ssm_state"].shape}')
            
            
            # print (f'[mamba_model.py]: qa_batch_input.shape: {qa_batch_input}')
        
            insert_states = True 
            retrieve_states = False  
            insert_states_for_training = True 
            # inserted_all_states = all_layers_states_dict 
            
            # Before the backward pass, retain gradients for the tensor
            # self.all_layers_states_dict[1]["ssm_state"][0].retain_grad()
            # print (f'[mamba_model.py]: retain_grad()')
            # qa_batch_input.retain_grad() 
            # print (f'[mamba_model.py]: qa_batch_input.retrain_grad() ')
            
            
            
            
            try: 
                # self.all_layers_states_dict[54]["ssm_state"].register_hook(lambda grad: self._report_self_total_grad_norm(grad, 'FIRST forward returned ssm-states at LAYER-54'))
                # all_soupped_states[54]["ssm_state"].register_hook(lambda grad: self._report_self_total_grad_norm(grad, 'SECOND forward inserted all_soupped_states at LAYER-54'))
                
                # self.all_layers_states_dict[55]["ssm_state"].register_hook(lambda grad: self._report_self_total_grad_norm(grad, 'FIRST forward returned ssm-states at LAYER-55'))
                # all_soupped_states[55]["ssm_state"].register_hook(lambda grad: self._report_self_total_grad_norm(grad, 'SECOND forward inserted all_soupped_states at LAYER-55'))
                
                # self.all_layers_states_dict[56]["ssm_state"].register_hook(lambda grad: self._report_self_total_grad_norm(grad, 'FIRST forward returned ssm-states at LAYER-56'))
                # all_soupped_states[56]["ssm_state"].register_hook(lambda grad: self._report_self_total_grad_norm(grad, 'SECOND forward inserted all_soupped_states at LAYER-56'))
                current_device = f"cuda:{hidden_states.device}"
                # self.all_layers_states_dict[54]["ssm_state"].register_hook(lambda grad: print (f'FIRST forward returned ssm-states at LAYER-54 has grad.shape: {grad.shape} \n FIRST-LAYER-54-grad[0]-{current_device}:{grad[0]} \n FIRST-LAYER-54-grad[1]-{current_device}:{grad[1]}'))
                # all_soupped_states[54]["ssm_state"].register_hook(lambda grad: print (f'SECOND forward inserted all_soupped_states at LAYER-54 has grad.shape: {grad.shape} \n SECOND-LAYER-54-grad[0]-{current_device}:{grad[0]}'))
                
                # self.all_layers_states_dict[55]["ssm_state"].register_hook(lambda grad: print (f'FIRST forward returned ssm-states at LAYER-55 has grad.shape: {grad.shape} \n FIRST-LAYER-55-grad[0]-{current_device}:{grad[0]} \n FIRST-LAYER-55-grad[1]-{current_device}:{grad[1]}'))
                # all_soupped_states[55]["ssm_state"].register_hook(lambda grad: print (f'SECOND forward inserted all_soupped_states at LAYER-55 has grad.shape: {grad.shape} \n SECOND-LAYER-55-grad[0]-{current_device}:{grad[0]}'))
                
                # self.all_layers_states_dict[56]["ssm_state"].register_hook(lambda grad: print (f'FIRST forward returned ssm-states at LAYER-56 has grad.shape: {grad.shape} \n FIRST-LAYER-56-grad[0]-{current_device}:{grad[0]} \n FIRST-LAYER-56-grad[1]-{current_device}:{grad[1]}'))
                # all_soupped_states[56]["ssm_state"].register_hook(lambda grad: print (f'SECOND forward inserted all_soupped_states at LAYER-56 has grad.shape: {grad.shape} \n SECOND-LAYER-56-grad[0]-{current_device}:{grad[0]}'))
            except: 
                print (f'[mamba_model.py]: failed to register hook to self.all_layers_states_dict or all_soupped_states. Could be due to evaluation mode.')
            
            # with torch.no_grad (): 
            print (f'[mamba_model.py]: Entering SECOND forward-pass')
            hidden_states, all_layers_states_dict_2 = self.decoder(
                                                        # Zixian: Oct 30: Inserting splitted QA embeddings 
                                                        # hidden_states=decoder_input,
                                                        hidden_states=qa_batch_input, 
                                                        
                                                        attention_mask=attention_mask,
                                                        inference_params=inference_params,
                                                        rotary_pos_emb=rotary_pos_emb,
                                                        
                                                        # Insert states params
                                                        insert_states=insert_states, 
                                                        retrieve_states=retrieve_states, 
                                                        
                                                        # Inserting soupped states 
                                                        inserted_all_states=all_soupped_states, 
                                                        
                                                        insert_states_for_training=insert_states_for_training, 
                                                        # Zixian: Oct 28: Not used in new version of Megatron Mamba
                                                        # **(extra_block_kwargs or {}),
                                                        )
            
            print (f'[mamba_model.py]: After SECOND forward-pass')
            # try: 
            #     qa_batch_input.register_hook(lambda grad: self._report_self_total_grad_norm(grad, 'qa_batch_input from SECOND forward'))
            # except: 
            #     print (f'[mamba_model.py]: failed to register hook to self.all_layers_states_dict or all_soupped_states. Could be due to evaluation mode.')

            
            
            # print (f'[mamba_model.py]: qa_batch_input.requires_grad: {qa_batch_input.requires_grad}')
            # print (f'[mamba_model.py]: qa_batch_input.grad: {qa_batch_input.grad}')
            
            # print (f'[mamba_model.py]: all_layers_states_dict [1]["ssm_state"][0].requires_grad: {all_layers_states_dict [1]["ssm_state"][0].requires_grad}')
            # print (f'[mamba_model.py]: all_layers_states_dict [1]["ssm_state"][0].grad: {all_layers_states_dict [1]["ssm_state"][0].grad}')
            
            
            
            
            # print (f'[mamba_model.py]: all_layers_states_dict [1]["ssm_state"][0].requires_grad: {self.all_layers_states_dict [1]["ssm_state"][0].requires_grad}')
            # print (f'[mamba_model.py]: all_layers_states_dict [1]["ssm_state"][0].grad: {self.all_layers_states_dict [1]["ssm_state"][0].grad}')

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
