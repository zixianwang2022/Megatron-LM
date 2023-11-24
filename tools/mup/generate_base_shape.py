"""Generate the base shape file.

The base shape file is used to record the shape of base models to compute
the multipliers, as well as indicating which variables are matrix-like
(lr and init scale with width) and vector-like (lr and init does not scale with width).
"""
import argparse
import sys
sys.path.append('/lustre/fsw/llmservice_nlp_fm/chzhu/sources/megatron-lm')

from megatron.arguments import (parse_args, validate_args)
from megatron.global_vars import set_global_variables
from megatron import get_args
from megatron.core.models.gpt import GPTModel
from megatron.core.enums import ModelType
from tools.mup.shape import make_base_shapes
from megatron.initialize import initialize_megatron
from megatron import print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    gpt_layer_with_transformer_engine_spec_moe
)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts is None:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
            else:
                transformer_layer_spec = gpt_layer_with_transformer_engine_spec_moe

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent
        )
    else:
        assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
    return model

def mup_parse_args(parser_main):
    # This part should include as many variables as possible that changes the shapes from smaller to larger models.
    parser = parser_main.add_argument_group(title='muP shape file generation')
    parser.add_argument(
        '--base-hidden-size', default=256, type=int, required=True,
        help='Hidden size of the base model.')
    parser.add_argument(
        '--base-num-attention-heads', default=8, type=int, required=True,
        help='Number of attention heads for the base model.')
    parser.add_argument(
        '--delta-hidden-size', default=512, type=int, required=True,
        help='Hidden size of the delta model.')
    parser.add_argument(
        '--delta-num-attention-heads', default=8, type=int, required=True,
        help='Number of attention heads for the delta model.')
    parser.add_argument(
        '--shape-file-path', default='', type=str,
        help='Path to store the generated shape file.')

    return parser_main


def get_model_from_configs(args, hidden_size, num_attention_heads, model_name='base model'):
    args.hidden_size = hidden_size
    args.num_attention_heads = num_attention_heads
    assert args.hidden_size % args.num_attention_heads == 0
    args.kv_channels = args.hidden_size // args.num_attention_heads
    args.ffn_hidden_size = 4 * args.hidden_size
    if args.swiglu:
        args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64

    config_base = core_transformer_config_from_args(args)

    # set_global_variables(args)
    print_rank_0(f"Args of {model_name}")
    print_rank_0("-----"*10)
    print_rank_0(args)
    model = model_provider()

    return model


def main():
    parser = argparse.ArgumentParser()
    mup_args, _ = mup_parse_args(parser).parse_known_args()

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=mup_parse_args,
                        args_defaults={})

    args = get_args()
    args.model_type = ModelType.encoder_or_decoder

    print(args.tensor_model_parallel_size)

    assert args.tensor_model_parallel_size == 1,\
        "When generating shape files, we should always set args.tensor_model_parallel_size == 1 to get the actual model size!"

    # for base model
    base_model = get_model_from_configs(
        args, hidden_size=mup_args.base_hidden_size,
        num_attention_heads=args.base_num_attention_heads, model_name='Base model')

    # for delta model
    delta_model = get_model_from_configs(
        args, hidden_size=mup_args.delta_hidden_size,
        num_attention_heads=args.delta_num_attention_heads, model_name='Delta model')

    make_base_shapes(base_model, delta_model, savefile=mup_args.shape_file_path)


if __name__ == '__main__':
    main()
