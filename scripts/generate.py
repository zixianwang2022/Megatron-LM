# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import abc
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple, TypedDict

import sys
sys.path.append("/lustre/fs1/portfolios/adlr/users/lmcafee/llama/2/llama")
from llama.generation import Llama, sample_top_p
from llama.tokenizer import Tokenizer as OriginalLlamaTokenizer

from megatron import get_args, get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config
from megatron.initialize import initialize_megatron, set_jit_fusion_options
from megatron.model import DistributedDataParallel as LocalDDP, Float16Module
from megatron.training import get_model as _get_model
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from pretrain_gpt import model_provider

# >>>
from lutil import pax, tp
# <<<

tokenizer_path = "/lustre/fs1/portfolios/adlr/users/lmcafee/llama/2/llama/tokenizer.model"

class Lab(abc.ABC):

    def __init__(self, model, tokenizer, pad_id):
        self.model = model
        self.tokenizer = tokenizer
        self.pad_id = pad_id

    def get_ntokens(self, tokens):
        assert tokens.shape
        # return torch.sum((tokens != self.tokenizer.eod).view(-1)).item()
        return torch.sum((tokens != self.pad_id).view(-1)).item()

    @abc.abstractmethod
    def _tokenize(self, text):
        pass

    def tokenize(self, text):

        args = get_args()

        tokens = torch.tensor(
            self._tokenize(text),
            dtype=torch.long,
            device=torch.cuda.current_device())
        tokens = torch.cat([
            tokens,
            torch.full(
                (args.seq_length - tokens.numel(),),
                self.pad_id, # self.tokenizer.eod,
                dtype=torch.long,
                device=torch.cuda.current_device(),
            ),
        ], dim=0)
        # tokens = tokens.reshape((1, -1)) # (args.micro_batch_size, -1))

        # pax({
        #     "text" : text,
        #     "tokens" : tp(tokens),
        #     "n_tokens" : self.get_ntokens(tokens),
        # })

        return tokens

    @abc.abstractmethod
    def _detokenize(self, tokens):
        pass

    def detokenize(self, tokens):

        args = get_args()

        assert tokens.shape == (args.seq_length,)

        n_tokens = self.get_ntokens(tokens)

        text = self._detokenize(tokens[:n_tokens].tolist())

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(text)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        pax({
            "n_tokens" : n_tokens,
            "tokens" : tp(tokens),
            "text" : text,
        })

        return text

    @abc.abstractmethod
    def forward(self, tokens):
        pass

class MegatronLab(Lab):

    @classmethod
    def get_model(cls, no_wd_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
        """Setup model and optimizer."""
        args = get_args()

        models = _get_model(model_provider, ModelType.encoder_or_decoder)
        unwrapped_model = unwrap_model(models, (torchDDP, LocalDDP, Float16Module))
        args.iteration = load_checkpoint(models, None, None)

        # pax({"models": models})

        return models[0]

    def __init__(self):

        if 1:
            tokenizer = get_tokenizer()
            super().__init__(self.get_model(), tokenizer, tokenizer.eod)
        else:
            tokenizer = OriginalLlamaTokenizer(model_path=tokenizer_path)
            # pax({
            #     "tokenizer" : tokenizer,
            #     "n_words" : tokenizer.n_words,
            #     # "eod_id" : tokenizer.eod_id,
            #     "pad_id" : tokenizer.pad_id,
            # })
            super().__init__(self.get_model(), tokenizer, tokenizer.pad_id)

        # self.config = get_model_config(self.model)
        # self.seq_length = self.model.config.seq_length

    def _tokenize(self, text):
        return self.tokenizer.tokenize(text, bos=True, eos=False)
    def _detokenize(self, tokens):
        return self.tokenizer.detokenize(tokens)

    # def _tokenize(self, text):
    #     return self.tokenizer.encode(text, bos=True, eos=False)
    # def _detokenize(self, tokens):
    #     return self.tokenizer.decode(tokens)

    def forward(self, tokens):

        args = get_args()

        # assert tokens.shape == (1, args.seq_length)
        assert tokens.shape == (args.seq_length,)

        # >>>
        # tokens = tokens[:self.get_ntokens(tokens)]
        # <<<
        tokens = tokens.reshape((1, -1))

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.pad_id, # self.tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tokens, position_ids, attention_mask)

        logits = logits[0]

        pax({
            "tokens" : tp(tokens),
            "logits" : tp(logits),
        })

        return logits

class LlamaLab(Lab):

    def __init__(self):

        args = get_args()
        # args.seq_length = 128

        generator = Llama.build(
            ckpt_dir="/lustre/fs1/portfolios/adlr/users/lmcafee/llama/2/llama/llama-2-7b",
            tokenizer_path=tokenizer_path,
            max_seq_len=args.seq_length,
            max_batch_size=1,
            model_parallel_size=None,
        )

        super().__init__(
            generator.model,
            generator.tokenizer,
            generator.tokenizer.pad_id,
        )

        # pax({"generator": generator})

    def _tokenize(self, text):
        return self.tokenizer.encode(text, bos=True, eos=False)

    def _detokenize(self, tokens):
        return self.tokenizer.decode(tokens)

    def forward(self, tokens):

        args = get_args()

        assert tokens.shape == (args.seq_length,)

        n_tokens = self.get_ntokens(tokens)

        tokens = tokens.reshape((1, -1))
        logits = self.model.forward(tokens[:, :n_tokens], 0)
        logits = logits[0]

        pax({
            "tokens" : tp(tokens),
            "logits" : tp(logits),
        })

        return logits

@torch.inference_mode()
def generate(
        lab: Lab,
        input_text: str,
        # max_output_len: int,
        # temperature: float = 0.6,
        temperature: float = 0.,
        top_p: float = 0.9,
        # logprobs: bool = False,
        # echo: bool = False,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:

    tokens = lab.tokenize(input_text)
    n_tokens = lab.get_ntokens(tokens)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(tokens[:n_tokens].tolist())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    pax({
        "input_text" : input_text,
        "tokens" : tp(tokens),
        "n_tokens" : n_tokens,
    })
    
    # tokens = tokens.reshape((1, -1))
    for i in tqdm(range(n_tokens, n_tokens + 50), "gen tokens"):

        logits = lab.forward(tokens)

        if temperature > 0:
            # probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            probs = torch.softmax(logits[-1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            # next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = torch.argmax(logits[-1], dim=-1)

        next_token = next_token.reshape(-1)
        tokens[i] = next_token

        # pax({
        #     "tokens" : tp(tokens),
        #     "logits" : tp(logits),
        #     "next_token" : next_token.item(),
        #     "n_tokens" : lab.get_ntokens(tokens),
        # })
    # tokens = tokens.reshape(-1)

    output_text = lab.detokenize(tokens)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(output_text)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    pax({
        "input_text" : input_text,
        "output_text" : output_text,
    })

def add_textgen_args(parser):
    group = parser.add_argument_group(title="Text generation.")
    group.add_argument("--gen-model", choices=["megatron","llama"], required=True)
    return parser

if __name__ == "__main__":

    # {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}

    # Initalize.
    initialize_megatron(extra_args_provider=add_textgen_args)
    set_jit_fusion_options()

    # Model, tokenizer.
    args = get_args()
    if args.gen_model == "megatron":
        lab = MegatronLab()
    elif args.gen_model == "llama":
        lab = LlamaLab()
    else:
        raise Exception("specialize for '%s'." % args.gen_model)

    input_text = "lawrence is the fastest cyclist since "
    # input_text = "the three most important inventions are "
    # input_text = "the most important thing nvidia did was "
    # input_text = "it just makes me so angry that "
    # input_text = "the funniest knock knock joke i ever heard was "
    # max_output_len = 64

    # Generate.
    output_text = generate(lab, input_text) # , max_output_len)

    pax({
        "tokenizer" : tokenizer,
        "text" : text,
        "tokens" : tp(tokens),
        "attention_mask" : tp(attention_mask),
        "loss_mask" : tp(loss_mask),
        "position_ids" : tp(position_ids),
        "output_tensor" : tp(output_tensor),
    })

    # return {'text': np.array(sample, dtype=np.int64)}
    


