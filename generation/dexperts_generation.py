# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT"""
import json
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel, ModelType, Float16Module
from megatron.training import get_model
from megatron.text_generation import dexpert_generate_and_post_process


def expert_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    print_rank_0('building expert GPT model ...')
    args.num_layers = 24
    args.hidden_size = 2048
    args.num_attention_heads = 32
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    expert = GPTModel(num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process)

    return expert

def antiexpert_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    print_rank_0('building antiexpert GPT model ...')
    args.num_layers = 24
    args.hidden_size = 2048
    args.num_attention_heads = 32
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    antiexpert = GPTModel(num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process)

    return antiexpert


def get_no_pipepine_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()
    add_encoder = True
    add_decoder = True
    if model_type == ModelType.encoder_and_decoder:
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            assert args.pipeline_model_parallel_split_rank is not None, \
                "Split rank needs to be specified for model with both encoder and decoder"
            rank = mpu.get_pipeline_model_parallel_rank()
            split_rank = args.pipeline_model_parallel_split_rank
            world_size = mpu.get_pipeline_model_parallel_world_size()
            pre_process = rank == 0 or rank == split_rank
            post_process = (rank == (split_rank - 1)) or (
                    rank == (world_size - 1))
            add_encoder = mpu.is_pipeline_stage_before_split()
            add_decoder = mpu.is_pipeline_stage_after_split()
        model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder)
    else:
        model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process
        )
    model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
            i = torch.cuda.current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
                     for model_module in model]

        elif args.DDP_impl == 'local':
            from megatron.model import DistributedDataParallel as LocalDDP
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]

        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))

    return model

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process)

    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')
    group.add_argument("--use_keywords", action='store_true',
                        help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs")
    group.add_argument("--decay_constant", type=float, default=50,
                        help="Value for the decay constant (lambda in the paper)")
    group.add_argument("--epsilon", type=float, default=0.01,
                        help="Minimum factor by which each probability is multiplied")
    group.add_argument("--debug", action='store_true',
                        help="If set, additional debugging output is printed to stdout")
    group.add_argument('--adaptor', action='store_true', default=False)
    group.add_argument('--project-size', type=int, default=256)
    # dexperts
    group.add_argument("--alpha", type=float, default=2.0,
                        help="Hyperparameter for dexperts")
    group.add_argument("--toxic-model", type=str, default='',
                        help="anti-expert model load path for dexperts")
    group.add_argument("--nontoxic-model", type=str, default='',
                        help="expert model load path for dexperts")
    return parser


def generate_samples_conditional(model, expert=None, antiexpert=None):
    args = get_args()

    if torch.distributed.get_rank() == 0:
        num_samples = args.num_samples
        cnt = 0
        from tqdm import tqdm
        pbar = tqdm(total=num_samples)

        fname = open(args.sample_input_file, "r")
        lines = fname.readlines()
        all_raw_text = [json.loads(line)['prompt']['text'] for line in lines]
        input_count = len(all_raw_text)
        input_pos = 0

    while True:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            sentences = []
            print("global batch size", args.global_batch_size)
            for _ in range(args.global_batch_size):
                if input_pos >= input_count:
                    print(f"input pos: {input_pos}, input count: {input_count}")
                    raw_text = "EMPTY TEXT"
                else:
                    raw_text = all_raw_text[input_pos]
                input_pos += 1
                sentences.append(raw_text)

            print_rank_0(sentences)
            max_len = args.out_seq_length
            resp_sentences, resp_sentences_seg, output_logits, \
            tokens = dexpert_generate_and_post_process(model, prompts=sentences,
                                               tokens_to_generate=max_len,
                                               return_output_log_probs=False,
                                               top_k_sampling=args.top_k,
                                               top_p_sampling=args.top_p,
                                               add_BOS=False,
                                               temperature=1.0,
                                               expert=expert, antiexpert=antiexpert)
            # print("len of tokens[0]", len(tokens[0]))
            # print(resp_sentences_seg[0])
            # print("len of tokens[1]", len(tokens[1]))
            for prompt, generation, token in zip(sentences, resp_sentences, tokens):
                datum = {'text': generation[len(prompt):], 'all_text': generation, 'prompt': prompt, 'id': cnt}
                # print_rank_0(("len of tokens", len(token)))
                # print_rank_0(datum)
                yield datum
                cnt += 1
                pbar.update()
                if cnt >= num_samples:
                    break

            if cnt >= num_samples:
                pbar.close()
                break
        else:
            dexpert_generate_and_post_process(model)



def generate_and_write_samples_conditional(model, expert=None, antiexpert=None):
    args = get_args()
    if args.sample_output_file is None:
        sample_output_file = args.sample_input_file + ".out"
        print('`sample-output-file` not specified, setting '
              'it to {}'.format(sample_output_file))
    else:
        sample_output_file = args.sample_output_file
    with open(sample_output_file, 'w') as f:
        for datum in generate_samples_conditional(model, expert=expert, antiexpert=antiexpert):
            if torch.distributed.get_rank() == 0:
                f.write(json.dumps(datum) + '\n')


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True,
                                       'seq_length': 2048})
    args = get_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    load = args.load

    args.num_layers = 24
    args.hidden_size = 2048
    args.num_attention_heads = 32
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1

    # Set up model and load checkpoint
    expert = get_model(expert_provider, wrap_with_ddp=False)
    args.load = args.nontoxic_model
    if args.load is not None:
        _ = load_checkpoint(expert, None, None)
    expert = expert[0]


    antiexpert = get_model(antiexpert_provider, wrap_with_ddp=False)
    args.load = args.toxic_model
    if args.load is not None:
        _ = load_checkpoint(antiexpert, None, None)
    antiexpert = antiexpert[0]

    args.num_layers = num_layers
    args.hidden_size = hidden_size
    args.num_attention_heads = num_attention_heads
    args.tensor_model_parallel_size = tensor_model_parallel_size
    args.pipeline_model_parallel_size = pipeline_model_parallel_size
    model = get_model(model_provider, wrap_with_ddp=False)
    args.load = load
    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    model = model[0]

    # Generate samples.
    if args.sample_input_file != None:
        print(f"{args.sample_input_file}")
        generate_and_write_samples_conditional(model, expert=expert, antiexpert=antiexpert)
    else:
        print("Unconditional generation currently not supported")


if __name__ == "__main__":

    main()
