# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
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
import numpy as np
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import FlamingoModel
from megatron.training import get_model
from megatron.text_generation.flamingo_api import flamingo_generate_and_post_process, flamingo_beam_search_and_post_process

def model_provider(pre_process=True, post_process=True):
	"""Build the model."""

	print_rank_0('building Flamingo model ...')
	model = FlamingoModel(
		num_tokentypes=0,
		parallel_output=True,
		pre_process=pre_process,
		post_process=post_process
	)
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
	group.add_argument("--dataset", type=str)
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
	group.add_argument('--add_retriever', action='store_true', default=False)
	group.add_argument('--adaptor', action='store_true', default=False)
	return parser

def generate_samples_unconditional(model):
	args = get_args()

	if torch.distributed.get_rank() == 0:
		cnt = 0
		num_samples = args.num_samples
		from tqdm import tqdm
		pbar = tqdm(total=num_samples)

	if (args.dataset == "COCO"):
		token_embs = np.load("/mnt/fsx-main/zhuoliny/karpathy/test_emb/img_emb/img_token_emb_0.npy")
		print(token_embs.shape)
		import pickle
		with open("/mnt/fsx-main/zhuoliny/karpathy/COCO_imageid.pkl", "rb") as tf:
			_train, _val, _test = pickle.load(tf)
		print(len(_val))
	elif (args.dataset == "NoCaps"):
		token_embs = np.load("/mnt/fsx-main/zhuoliny/nocaps/test_emb/img_emb/img_token_emb_0.npy")
		print(token_embs.shape)
		_test = [i for i in range(token_embs.shape[0])]
	while True:
		if torch.distributed.get_rank() == 0:
			sentences = [''] * args.global_batch_size
			print("global batch size", args.global_batch_size)
			max_len = args.out_seq_length

			resp_sentences, resp_sentences_seg, output_logits, \
			tokens = flamingo_generate_and_post_process(model, prompts=sentences,
											   tokens_to_generate=max_len,
											   return_output_log_probs=False,
											   top_k_sampling=args.top_k,
											   top_p_sampling=args.top_p,
											   add_BOS=True,
											   temperature=1.0, 
											   vision_inputs=token_embs[cnt])
			resp_sentences[0] = resp_sentences[0].replace("<|endoftext|> ","").replace("<EOC>","").replace("<|endoftext|>", "")
			for prompt, generation in zip(sentences, resp_sentences):
				datum = {"image_id": _test[cnt], "caption": generation[len(prompt):]}#, 'all_text': generation, 'prompt': prompt, 'id': cnt}
				yield datum
				cnt += 1
				pbar.update()
				if cnt >= num_samples:
					break

			if cnt >= num_samples:
				pbar.close()
				break
		else:
			flamingo_generate_and_post_process(model)



def generate_and_write_samples_unconditional(model):
	args = get_args()
	assert args.genfile is not None
	with open(args.genfile, 'w') as f:
		for datum in generate_samples_unconditional(model):
			if torch.distributed.get_rank() == 0:
				f.write(json.dumps(datum) + '\n')


def main():
	"""Main program."""

	initialize_megatron(extra_args_provider=add_text_generate_args,
						args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
									   'no_load_rng': True,
									   'no_load_optim': True,
									   'seq_length': 256})

	# Set up model and load checkpoint
	model = get_model(model_provider, wrap_with_ddp=False)

	args = get_args()

	if args.load is not None:
		_ = load_checkpoint(model, None, None)
	model = model[0]

	generate_and_write_samples_unconditional(model)


if __name__ == "__main__":

	main()
