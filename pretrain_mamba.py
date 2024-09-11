# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from torch import Tensor
from functools import partial

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
# from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


stimer = StragglerDetector()

def count_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            num_params += param.numel()
            print_rank_0(f" - {name}: {param.numel()}")
    return num_params


def freeze_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = False 
            print_rank_0(f" - {name}: param.requires_grad = {param.requires_grad}")
    return num_params




def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
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
        
        # freeze_parameters_in_layer(model, f'decoder.layers.{l}.')
        
    
    # Freeze the entire Mamba model except for decoder output linear layer 
    # model.freeze (freeze_mamba_model=args.freeze_mamba_blocks) 
    model.freeze (freeze_mamba_model=True, freeze_embedding_model=True, freeze_output_layer=False) 
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()

def loss_func(loss_mask: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()
    
    print_rank_0 (f'\n\nforward_step data_iterator:\n{data_iterator}\n\n\n')

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()
    
    
    tokenizer = get_tokenizer()
    
    print_rank_0 (f'\n\n tokens:\n{tokens}\n\n\n')
    print_rank_0 (f'\n\n tokens:\n{tokens[0][0]}\n\n\n')
    print_rank_0 (f'\n\n tokens:\n{type (int (tokens[0][0]))}\n\n\n')
    print_rank_0 (f'\n\n decoded tokens: {tokenizer.detokenize (int(tokens[0][0]))}')
    print_rank_0 (f'{tokenizer.detokenize (int(tokens[0][1]))}')
    print_rank_0 (f'{tokenizer.detokenize (int(tokens[0][2]))}')
    
    # print_rank_0 (f'\n\n tokens:\n{tokens}\n\n\n')
    # print_rank_0 (f'\n\n labels:\n{labels}\n\n\n')

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def add_mamba_states_args(parser):
    group = parser.add_argument_group(title='mamba states')
    # Inference
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
    
    # Training: 
    # Insert states
    group.add_argument ("--insert_mamba_states_for_training", type=bool, default=False, 
                        help="Whether to insert mamba hidden states for training.")
    group.add_argument ("--insert_mamba_states_for_training_dir", type=str, default=None, 
                        help="The directory of the mamba hidden states pickle files that will be used for training.")
    # Retrieve states for training during inference 
    group.add_argument ("--retrieve_mamba_states_for_training", type=bool, default=False, 
                        help="Whether to retrieve mamba hidden states during inference that will be used later for training.")
    group.add_argument ("--retrieve_mamba_states_for_training_dir", type=str, default=None, 
                        help="The directory of the mamba hidden states pickle files will be stored that will be used for training.")
    group.add_argument ("--retrieve_mamba_states_for_training_filename", type=str, default=None, 
                        help="The filename for the currently retrieving states will be named")
    
    
    
    
    return parser


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}, 
             extra_args_provider=add_mamba_states_args)
