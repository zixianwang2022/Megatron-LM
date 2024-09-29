# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Literal, Optional, Union

import os 
import torch
from torch import Tensor
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

class MambaModel(LanguageModule):
    """Mamba language model.

    Args:
        config (TransformerConfig): Transformer config
        mamba_stack_spec (ModuleSpec): Specifies the modules to use for the various layer types
        attention_layer_spec (ModuleSpec): Specifies module to use for attention layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        hybrid_attention_ratio (float, optional): Specifies the ratio of attention layers to total layers.
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

        self.mamba_stack_spec: ModuleSpec = mamba_stack_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.hybrid_override_pattern = hybrid_override_pattern
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type=position_embedding_type
        
        self.inserted_all_states = None 

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type
            )

        if self.position_embedding_type == 'rope':
            
            print (f'\n\n\n INCLUDING positional embedding in MambaModel \n\n\n')
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        self.decoder = build_module(
            mamba_stack_spec,
            self.config,
            pre_process=self.pre_process,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            post_process=self.post_process,
            # self.vocab_size,
            # device="cuda",
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
        
        
        
    
    def set_input_states(self, input_states: dict) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        
        # if input_states is not None:
        #     assert len(input_states.keys()) == 1, 'input_states first key should only be length 1'
        
        self.inserted_all_states = input_states 
        print (f' \n\n Setting self.inserted_all_states = {input_states}  \n\n')
        # self.decoder.set_input_states (self.inserted_all_states)
        
        
    
    def combine_list_states_to_batch (self, list_of_states): 
        """ 
        list_of_states: a list of states dict
        
        return: 1 states dict where the ssm & conv states for each layer is batched in the order of the list
        
        Verified correctness with for both conv_state & ssm_state: 
        p = '/data/ziw081/ssm-retrieval/data/hotpot/training_data/100-data/hidden_states/gold-1/'
        s1 = torch.load (p+'877_251541_96202_467_202186_42839_625_75810_300_375_71763_724_17799_14_251519_251541_3_3_3_3_3_3_3_3_3_3_.pkl')
        s2 = torch.load (p+'877_251541_852_1703_339_276_6995_1586_49709_300_194695_6915_251597_14_251519_251541_3_3_3_3_3_3_3_3_3_3_.pkl')
        s3 = torch.load (p+'877_251541_796_6474_10956_42839_291_403_49236_251597_14_251519_251541_3_3_3_3_3_3_3_3_3_3_.pkl')
        sl = [s1, s2, s3]

        print (s1.keys())
        print (s1[0].keys())
        print (s1[0][55].keys())
        print (s1[0][55]['conv_state'].shape)


        o = combine_list_states_to_batch (sl)

        print (o.keys())
        print (o[0].keys())
        print (o[0][55].keys())
        print (o[0][55]['conv_state'].shape)

        print (o[0][55]['conv_state'][0])
        print (o[0][55]['conv_state'][1])
        print (o[0][55]['conv_state'][2])


        print (sl[0][0][55]['conv_state'])
        print (sl[1][0][55]['conv_state'])
        print (sl[2][0][55]['conv_state'])
        """
        
        output_all_states = {0:{}}
        # For every layer of the first element's starting position 
        for layer in list_of_states[0][0]: 
            
            
            batched_ssm_state = torch.cat (
                [each_state[0][layer]['ssm_state'] for each_state in list_of_states], dim=0
            )
            
            batched_conv_state = torch.cat (
                [each_state[0][layer]['conv_state'] for each_state in list_of_states], dim=0
            )
            
            output_all_states [0][layer]  = {'conv_state' : batched_conv_state, 'ssm_state':batched_ssm_state}
            
        return output_all_states
        
        
    def freeze(
        self, 
        freeze_mamba_model: bool, 
        freeze_embedding_model:bool, 
        freeze_output_layer:bool, 
        unfreeze_decoder_last_layer: bool = False,
    ):
        """
        Zixian: Sept 8 19:11pm VERIFIED functionality 
        
        Freeze model modules. 

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
        freeze_mamba_model (bool): Freeze the entire decoder module.
        freeze_embedding_model (bool): Freeze the embedding module.
        freeze_output_layer (bool): Freeze the output layer.
        unfreeze_decoder_last_layer (bool): Unfreeze decoder's last layer. 
        
        """
        
        # for l in range(self.model.decoder.num_layers_per_pipeline_rank):
        #     layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        modules = []
        if freeze_mamba_model:
            modules.append(self.decoder)
        if freeze_embedding_model:
            # Zixian: Prevent second host triggerring errors for hosting model only 
            if self.pre_process:
                modules.append(self.embedding)
        if freeze_output_layer:
            # Zixian: Prevent second host triggerring errors for hosting model only 
            if self.post_process:
                modules.append(self.output_layer)
        
        # Update Sept 7 22:12pm Not tested yet. 
        # TODO: if does not work, follow similar method in counting params 
        for module in modules:
            print (f' \n\n freezing {module} \n\n')
            for param in module.parameters():
                param.requires_grad = False
                
                
        # Freeze all decoder layers except the last one
        if unfreeze_decoder_last_layer:
            num_layers = len(self.decoder.layers)
            for idx, layer in enumerate(self.decoder.layers):
                if idx == num_layers - 1:
                    print(f'Unfreezing decoder layer {idx}')
                    for param in layer.parameters():
                        param.requires_grad = True 
                
                
 
        # for module in [self.decoder, self.embedding, self.output_layer]:
        #     for name, param in module.named_parameters():
        #         print(f"Module: {module.__class__.__name__}, Parameter: {name}, dtype: {param.dtype}")
 
        # from collections import Counter

        # dtype_counter = Counter()

        # for module in [self.decoder, self.embedding, self.output_layer]:
        #     for name, param in module.named_parameters():
        #         dtype = str(param.dtype)
        #         num_params = param.numel()
        #         dtype_counter[dtype] += num_params

        # # Print the counts
        # for dtype, count in dtype_counter.items():
        #     print(f"Data Type: {dtype}, Total Parameters: {count}")               
        

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        extra_block_kwargs: dict = None,
        
        # For retrieving states
        insert_states: bool =False, 
        retrieve_states: bool =False, 
        insert_states_for_training: bool = False, 
        inserted_all_states: Tensor =None, 
    ) -> Tensor:
        """Forward function of the Mamba model. This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units
        """
        # print ("Printing from Megatron-LM/megatron/core/ssm/mamba_model.py FUNC=forward line 165")
        # print (f'--insert_states:{insert_states}')
        # print (f'--retrieve_states:{retrieve_states}')

        # print (f"Printing from Megatron-LM/megatron/core/models/mamba/mamba_model.py Line 160")
        # print (f"--input_ids=\n{input_ids}")
        
        
        print_rank_0 (f'--mamba_model forward\n\n')
        # print_rank_0 (f'len (input_ids[0]):\n{len (input_ids[0])}\n\n')
        # print_rank_0 (f'len (input_ids[1]):\n{len (input_ids[1])}\n\n')
        print_rank_0 (f'input_ids:\n{input_ids}\n\n')
        print_rank_0 (f'decoder_input:\n{decoder_input}')
        
        
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        args = get_args()
        insert_states = args.inserting_mamba_states
        retrieve_states = args.retrieving_mamba_states
        insert_states_for_training = args.insert_mamba_states_for_training
        
        # Zixian: Sept 11 01:35am: 
        # Not sure if this will work for training 
        # Prevent inserting states when steping each time. 
        
        
        if input_ids is not None: 
            from megatron.training import get_tokenizer
            tokenizer = get_tokenizer ()
            a = tokenizer.detokenize (input_ids[0].tolist())
            print (f' \n\n input_ids.shape: {input_ids.shape} \n\n')
            print (f' \n\n input_ids[0].shape: {input_ids[0].shape} \n\n')
            print (f' \n\n input_ids[0].tolist(): {input_ids[0].tolist()} \n\n')
            print (f'\n\n decoded input_ids is: \n{a}\n')
        
        
        print (f' Checking whether look for states at mamba_model.py')
        
        if (input_ids is not None) and (len (input_ids[0]) > 1): 
            
            print (f' input_ids is not None, setting input states ')
            print (f'input_ids: \n{input_ids}')
        
            if (insert_states): 
                
                
                if args.insert_mamba_states_for_training: 
                    base_dir = os.path.dirname (args.insert_mamba_states_for_training_dir)
                    
                    # candidate_filename_tokens = input_ids[0]
                    print (f'input_ids.shape: {input_ids.shape}')
                    
                    # Initialize to collect each batch's loaded states 
                    list_of_batched_states = []
                    
                    for candidate_filename_tokens in input_ids: 
                        # candidate_filename = "".join ([f"{t}_" for t in candidate_filename_tokens])
                        
                        # Find the index of the last non-zero element
                        index = 0
                        while (candidate_filename_tokens[index] != 0) & ((candidate_filename_tokens[index] != 251519) | (candidate_filename_tokens[index+1] != 251541)):
                            index += 1
                        
                        print (f'index: {index}')
                        # if len (candidate_filename_tokens) > 25: 
                        if index > 25: 
                            candidate_filename = "".join ([f"{t}_" for t in candidate_filename_tokens[:25]])
                        else:
                            candidate_filename = "".join ([f"{t}_" for t in candidate_filename_tokens[:index]])
                        
                        matching_files = [filename for filename in os.listdir(base_dir) if candidate_filename in filename]
                        filename = ''
                        
                        # Check if there is at least one matching file and print it
                        if len (matching_files) == 1:
                            print("\n\n\n Found file:\n", matching_files[0])
                            print ('\n\n\n')
                            filename = matching_files[0]
                        
                        # Zixian: TODO: Raise error if can't find or conflicting states.pkl of the question input_ids. 
                        #               Think of other ways to continue training maybe. 
                        else:
                            
                            
                            print("No file found containing the substring.")
                            print (f'input_ids: {input_ids}')
                            print (f'candidate_filename: {candidate_filename}')
                            print (f'matching_files: {matching_files}')
                            if len (matching_files) == 0: 
                                raise (RuntimeError, f"Man, I can't find the corresponding states . pkl for this prompt: \n{input_ids}")
                            if len (matching_files) > 1: 
                                raise (RuntimeError, f"Man, I have found more than 1 ({len (matching_files)}) corresponding states . pkl s for this prompt: \n{input_ids}")
                            
                        filename = os.path.join (base_dir, filename)
                        self.inserted_all_states = torch.load (filename)
                        
                        list_of_batched_states.append (self.inserted_all_states)
                    
                    # Zixian: Sept 17 22:20 
                    # batch into 1 state_dict 
                    self.inserted_all_states = self.combine_list_states_to_batch (list_of_batched_states) 
                else: 
                    # print (f"----Loading self.inserted_all_states")
                    # Need to unpickle from the states pickle path stored in args 
                    self.inserted_all_states = torch.load (args.inserted_mamba_states_path)
                    # print (f"----Successfully loaded self.inserted_all_states")
        
            print (f' \n\n inside mamba_model.py: printing decoded input_ids: ')
            from megatron.training import get_tokenizer
            tokenizer = get_tokenizer ()
            a = tokenizer.detokenize (input_ids[0].tolist())
            # print (f'\n input_ids : a: {input_ids[0]} \n')
            print (f'\n detokenized : input_ids: {a} \n')
        
        
        
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
        args.global_counter_cnt += 1 
        # print (f'GLOBAL_CNT={args.global_counter_cnt} at Megatron-LM/megatron/core/models/mamba/mamba_model.py FUNC=forward line 191')
    

        # decoder_input = torch.ones(2, 2, dtype=torch.long)
        # print (f"Printing from Megatron-LM/megatron/core/models/mamba/mamba_model.py Line 191")
        # print (f"--decoder_input=\n{decoder_input}")
        
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
        print (f'At mamba_model.py, pipeline_rank={pipeline_rank} \n self.inserted_all_states is :\n {self.inserted_all_states}')
        print (f'decoder_input is: \n{decoder_input}')
        print (f'input_ids is: \n{input_ids}')
        
        
        # # Pass states through communication to other gpus 
        # self.decoder.set_input_states(self.inserted_all_states)
        
        
        # Run decoder.
        hidden_states, all_layers_states_dict = self.decoder(
                                                hidden_states=decoder_input,
                                                attention_mask=attention_mask,
                                                inference_params=inference_params,
                                                rotary_pos_emb=rotary_pos_emb,
                                                insert_states=insert_states, 
                                                retrieve_states=retrieve_states, 
                                                inserted_all_states=self.inserted_all_states, 
                                                insert_states_for_training=insert_states_for_training, 
                                                **(extra_block_kwargs or {}),
        )
        
        
        args.global_counter_cnt += 1 
        # print (f'GLOBAL_CNT={args.global_counter_cnt} at Megatron-LM/megatron/core/models/mamba/mamba_model.py FUNC=forward line 210')
    
        if parallel_state.is_pipeline_first_stage():
            import random 
            self.inserted_all_states = {'A': random.randint (10,99), 'B': random.randint (10,99)}

        if not self.post_process:
            print (f'\n\n --returning from "not self.post_process \n\n')
            return hidden_states , self.inserted_all_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        # Going through output layer to get logits 
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # --------------------------------------------------------------------------------------------------------------
            # ---------------------------- Forward is returning from here when calling generate ----------------------------
            # --------------------------------------------------------------------------------------------------------------
            # print (f'--returning from "labels is None"')
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous(), all_layers_states_dict # returning model's states 

        loss = self.compute_language_model_loss(labels, logits)

        # print (f'--returning from "everything else"')
        print (f'\n\n --returning from loss \n\n')
        
        # Zixian Sept 28: 
        # Return self.inserted_all_states here to prevent downstream 
        # returned value splits loss, which will cause error 
        return loss, self.inserted_all_states