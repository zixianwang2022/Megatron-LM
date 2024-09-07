# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate Mamba"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_model
from megatron.training.arguments import core_transformer_config_from_args
from megatron.inference.text_generation_server import MegatronServer
from megatron.inference.text_generation import generate_and_post_process
from megatron.inference.text_generation import beam_search_and_post_process

import torch

def count_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            num_params += param.numel()
            print_rank_0(f" - {name}: {param.numel()}")
    return num_params

# Taken from pretrain_mamba.py
def model_provider(pre_process=True, post_process=True) -> MambaModel:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore Mamba model and if not the legacy Mamba model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Mamba: The returned model
    """
    args = get_args()

    print_rank_0('building Mamba model ...')
    config = core_transformer_config_from_args(get_args())

    if args.use_mcore_models:
        if args.spec is not None:
            mamba_stack_spec = import_module(args.spec)
        else:
            raise("You must provide a valid Mamba layer spec!")

        model = MambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            hybrid_attention_ratio=args.hybrid_attention_ratio,
            hybrid_mlp_ratio=args.hybrid_mlp_ratio,
            hybrid_override_pattern=args.hybrid_override_pattern,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type
        )
    else:
        raise("Mamba only supported in Mcore!")

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')
    group.add_argument("--inserting_mamba_states", type=bool, default=False,
                       help='Whether to insert mamba hidden states. Yes, then mamba_states flag is required')
    group.add_argument("--retrieving_mamba_states", type=bool, default=False,
                       help='Whether to retrieve mamba hidden states. Yes, then return list will have one more column')
    group.add_argument("--inserted_mamba_states_path", type=str, default=None,
                       help='A path to the hidden states pickle file that will be inserted into mamba')
    group.add_argument("--retrieved_mamba_states_path", type=str, default=None,
                       help='A path to the hidden states pickle file that the retrieved states will be stored')
    group.add_argument("--global_counter_cnt", type=int, default=0,
                       help='A global counter to track every function call step')
    
    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")
    args.exit_on_missing_checkpoint = True
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        print (f"printing from Megatron-LM/tools/run_mamba_text_generation_server.py line 116")
        print (f"--Trying to initialize Megatron Server")
        server = MegatronServer(model)
        print (f"--After initialize Megatron Server")
        print (f"--Trying to run Megatron Server")
        
        args.global_counter_cnt += 1 
        print (f'GLOBAL_CNT={args.global_counter_cnt} at Megatron-LM/tools/run_mamba_text_generation_server.py line 124')
        
        
        server.run("0.0.0.0",port=args.port)
        print (f"--After running Megatron Server")

    while True:
        print (f"printing from Megatron-LM/tools/run_mamba_text_generation_server.py line 121\n")
        choice = torch.tensor(1, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        if choice.item() == 0:
            try:
                print (f"printing from Megatron-LM/tools/run_mamba_text_generation_server.py line 126\n")
                generate_and_post_process(model)
            except ValueError as ve:
                pass
        elif choice.item() == 1:
            try:
                beam_search_and_post_process(model)
            except ValueError as ve:
                pass
