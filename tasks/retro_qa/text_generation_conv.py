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
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.core.tensor_parallel.data import get_tensor_model_parallel_src_rank, get_tensor_model_parallel_group
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation import generate_and_post_process, beam_search_and_post_process
#from finetune_gpt_with_pretrain import get_tasks_args
from main import get_tasks_args
from dataset import reformat_prompt_v2, preprocess
from dataset import load_incontext_fewshot_samples, reformat_prompt_with_fewshot_samples
import time
# from tasks.prompt_learning.task_datasets import e2e_format_query, xsum_format_s

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()
    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process)

    return model

def add_text_generate_args(parser):
    """Text generation arguments."""

    parser = get_tasks_args(parser)
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=256,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--fewshot-input-file", type=str, default=None,
                       help='Get in-context few-shot input from file')
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
    group.add_argument("--epsilon", type=float, default=0.01,
                        help="Minimum factor by which each probability is multiplied")
    group.add_argument("--debug-gen", action='store_true',
                        help="If set, additional debugging output is printed to stdout")
    
    # group.add_argument('--adaptor', action='store_true', default=False)
    # group.add_argument('--project-size', type=int, default=256)
    group.add_argument('--beam-search', action='store_true', help='activate beam search')
    group.add_argument('--beam-size', type=int, default=5,
                        help='beam size for beam search,')
    group.add_argument('--length-penalty', type=float, default=1.0,
                        help='length penalty')
    group.add_argument('--gen-start-idx', type=int, default=0,
                        help='project size for adapters')
    group.add_argument('--num-gen', type=int, default=-1,
                        help='project size for adapters')
    group.add_argument('--ckpt-step', type=int, default=None,
                        help='setting ckpt step manually')
    group.add_argument("--use-retrieved-neighbours", action='store_true', default=False,
                       help='Use retrieved neighbours')
    
    # in-context few-shot
    group.add_argument("--incontext-fewshot", default=False, action='store_true', help="use in-context few-shot")
    group.add_argument("--n-shot", type=int, default=None, help='number of fewshot samples')

    return parser

def generate_samples_conditional(model):
    args = get_args()
    start = time.time()
    avg_time = []
    tokenizer = get_tokenizer()
    model.eval()
    if torch.distributed.get_rank() == 0:

        data = preprocess(args.sample_input_file, inference_only=True, retrieved_neighbours=args.use_retrieved_neighbours)
        print("total rows {}".format(len(data)))
        all_data = data[args.gen_start_idx:] ## start fron gen_start_idx
        if args.num_gen > 0:
            all_data = all_data[:args.num_gen]
        input_count = len(all_data)
        input_pos = 0

        if args.incontext_fewshot:
            # load incontext fewshot samples
            fewshot_list = load_incontext_fewshot_samples(args.fewshot_input_file, args.n_shot)

    if args.beam_search:
        assert args.micro_batch_size == 1

    terminate_runs = 0
    while True:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            sentences = []
            n_arrays = []
            print("global batch size", args.global_batch_size)
            for _ in range(args.global_batch_size):
                print(input_pos)
                if input_pos >= input_count:
                    print("reach the last row")
                    break
                else:
                    sample = all_data[input_pos]
                input_pos += 1

                valid_tasks = ['nq', 'tqa', 'benz', 'landrover', 'ford', 'att', 'iternal', 'carmanual', 'nvit', 'tcs', 'sandia', 'dropbox']
                if args.task.lower() in valid_tasks or any([x in args.task.lower() for x in valid_tasks]):
                    max_target_len = args.out_seq_length
                    # disable it for GPT for now
                    # neighbours_array = pad_neighbours_for_query_only(args, [tokenizer.tokenize(neighbour) for neighbour in neighbours], tokenizer.eod, args.ft_neighbours)
                    query, _, neighbours = sample
                    tokenizer = get_tokenizer()

                    if args.incontext_fewshot:
                        input_tokens = reformat_prompt_with_fewshot_samples(query, neighbours, args.task, args.ft_neighbours, fewshot_list, max_target_len, tokenizer, args.seq_length)
                    else:
                        input_tokens = reformat_prompt_v2(query, neighbours, args.task, args.ft_neighbours, max_target_len, tokenizer, args.seq_length)
                    # input_tokens = reformat_prompt_v1(query, neighbours, args.task, args.ft_neighbours, max_target_len, tokenizer, args.seq_length)
                    print(input_tokens)
                    raw_text = tokenizer.detokenize(input_tokens)
                    print(raw_text)
                else:
                    raise ValueError("invalid arg for task")
                sentences.append(raw_text)
                # n_arrays.append(neighbours_array)
            # neighbours_array = np.array(n_arrays)

            if args.beam_search:
                neighbours_array = neighbours_array.repeat(args.beam_size, axis=0)
                resp_sentences, resp_sentences_seg, scores = \
                        beam_search_and_post_process(model, prompts=sentences,
                                                     neighbours_array=neighbours_array,
                                                     length_penalty=args.length_penalty,
                                                     tokens_to_generate=args.seq_length-args.m,
                                                     beam_size=args.beam_size,
                                                     add_BOS=False)
            else:
                resp_sentences, resp_sentences_seg, scores, \
                tokens = generate_and_post_process(model, prompts=sentences,
                                                   tokens_to_generate=args.out_seq_length - 2,
                                                   return_output_log_probs=False,
                                                   top_k_sampling=args.top_k,
                                                   top_p_sampling=args.top_p,
                                                   add_BOS=False,
                                                   temperature=1.0)
                # neighbours_array=neighbours_array, if retro
            # print("len of tokens[0]", len(tokens[0]))
            # print(resp_sentences_seg[0])
            print("len of resp_sentences", len(resp_sentences))
            # print("len of scores", len(scores))
            # print("scores", scores)
            # exit(0)
            for prompt, generation in zip(sentences, resp_sentences):
                # datum = generation[len(prompt):].replace("<|endoftext|>", "").strip()
                datum = generation[len(prompt):]
                if "<|endoftext|>" in datum:
                    datum = datum[:datum.find("<|endoftext|>")].strip()
                #if "\n\n" in datum:
                #    datum = datum.split("\n\n", 1)[0]
                datum = datum.replace("\n", " ")
                # print("len of tokens", len(token))
                print(datum)
                yield datum
            avg_time.append((time.time() - start) / args.global_batch_size)
            print("avg time for each sample: ", sum(avg_time) / len(avg_time))
            start = time.time()
            if input_pos >= input_count:
                print("finish all lines")
                terminate_runs = 1
        else:
            if args.beam_search:
                beam_search_and_post_process(model)
            else:
                generate_and_post_process(model)

        terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
        torch.distributed.broadcast(terminate_runs_tensor, 0)
        terminate_runs = terminate_runs_tensor[0].item()

        if terminate_runs == 1:
            return

def generate_and_write_samples_conditional(model):
    args = get_args()
    if args.sample_output_file is None:
        sample_output_file = args.sample_input_file + ".out"
        print('`sample-output-file` not specified, setting '
              'it to {}'.format(sample_output_file))
    else:
        sample_output_file = args.sample_output_file
    with open(sample_output_file, 'w') as f:
        for datum in generate_samples_conditional(model):
            if torch.distributed.get_rank() == 0:
                f.write(datum + '\n')


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    print(model)
    args = get_args()

    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    model = model[0]

    # Generate samples.
    if args.sample_input_file != None:
        print(f"{args.sample_input_file}")
        generate_and_write_samples_conditional(model)
    else:
        generate_and_write_samples_unconditional(model)


if __name__ == "__main__":

    main()
