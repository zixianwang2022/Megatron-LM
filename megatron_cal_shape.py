import argparse
from megatron.arguments import (parse_args, validate_args)
from megatron.global_vars import set_global_variables
import yaml
from megatron import get_args
from megatron.model.gpt_model import GPTModel
from megatron.core.enums import ModelType
from tools.mup.shape import make_base_shapes
from megatron.initialize import initialize_megatron
from megatron import print_rank_0
from megatron.arguments import core_transformer_config_from_args

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def muT_parse_args(parser_main):
    parser = parser_main.add_argument_group(title='phf-evaluate-tasks')
    parser.add_argument("--muT_config_file", default=None, required=True)
    parser.add_argument("--shape_file", default="")

    return parser_main

def main():
    parser = argparse.ArgumentParser()
    muT_args, unknown_args = muT_parse_args(parser).parse_known_args()
    muT_config = yaml.full_load(open(muT_args.muT_config_file, 'r'))

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=muT_parse_args,
                        args_defaults={})
    # for base model
    args = get_args()
    args.model_type = ModelType.encoder_or_decoder

    args.hidden_size = muT_config['base_model'].get('hidden-size')
    args.num_attention_heads = muT_config['base_model'].get('num-attention-heads')
    assert args.hidden_size % args.num_attention_heads == 0
    args.kv_channels = args.hidden_size // args.num_attention_heads
    args.ffn_hidden_size = 4 * args.hidden_size
    if args.swiglu:
        args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64

    config_base = core_transformer_config_from_args(args)

    # set_global_variables(args)
    print_rank_0("base model args")
    print_rank_0("-----"*10)
    print_rank_0(args)
    base_model = GPTModel(
        config_base,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True
    )

    # for delta model
    args.hidden_size = muT_config['delta_model'].get('hidden-size')
    args.num_attention_heads = muT_config['delta_model'].get('num-attention-heads')

    assert args.hidden_size % args.num_attention_heads == 0
    args.kv_channels = args.hidden_size // args.num_attention_heads
    args.ffn_hidden_size = 4 * args.hidden_size
    if args.swiglu:
        args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64

    print_rank_0("delta model args")
    print_rank_0("-----"*10)
    print(args)
    config_delta = core_transformer_config_from_args(args)
    # set_global_variables(args)
    delta_model = GPTModel(
        config_delta,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True
    )

    make_base_shapes(base_model, delta_model, savefile=muT_args.shape_file)


if __name__ == '__main__':
    main()
