# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple, TypedDict

import sys
sys.path.append("/lustre/fs1/portfolios/adlr/users/lmcafee/llama/2/llama")
from llama.generation import sample_top_p

from megatron import get_args, get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config
from megatron.initialize import initialize_megatron, set_jit_fusion_options
from megatron.model import DistributedDataParallel as LocalDDP, Float16Module
# from megatron.training import setup_model_and_optimizer
from megatron.training import get_model as _get_model
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from pretrain_gpt import model_provider

# >>>
from lutil import pax, tp
# <<<

def get_model(no_wd_decay_cond=None,
              scale_lr_cond=None,
              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()

    models = _get_model(model_provider, ModelType.encoder_or_decoder)
    unwrapped_model = unwrap_model(models, (torchDDP, LocalDDP, Float16Module))
    args.iteration = load_checkpoint(models, None, None)

    # pax({"models": models})

    return models[0]

def get_tokens(tokenizer, text):

    args = get_args()

    tokens = torch.tensor(
        tokenizer.tokenize(text, bos=True, eos=False),
        dtype=torch.long,
        device=torch.cuda.current_device())
    tokens = torch.cat([
        tokens,
        torch.full(
            (args.seq_length - tokens.numel(),),
            tokenizer.eod,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        ),
    ], dim=0)
    tokens = tokens.reshape((1, -1)) # (args.micro_batch_size, -1))

    return tokens

def get_ntokens(tokens):
    return torch.sum((tokens != tokenizer.eod).view(-1)).item()

def get_text(tokenizer, tokens):

    assert tokens.shape == (args.seq_length,)

    n_tokens = get_ntokens(tokens)

    text = tokenizer.detokenize(tokens.tolist())

    pax({
        "tokens" : tp(tokens),
        "n_tokens" : get_ntokens(tokens),
        "text" : text,
    })

# def get_input_args(tokenizer, text):

#     # Get the masks and postition ids.
#     attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
#         tokens,
#         tokenizer.eod,
#         args.reset_position_ids,
#         args.reset_attention_mask,
#         args.eod_mask_loss)

#     # Input args.
#     args = {
#         "text" : text,
#         "tokens" : tokens,
#         "attention_mask" : attention_mask,
#         "loss_mask" : loss_mask,
#         "position_ids" : position_ids,
#     }

#     # >>>
#     # pax({"args": args})
#     # <<<

#     return args

@torch.inference_mode()
def generate(
        model,
        tokenizer,
        input_text: str,
        max_output_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:

    tokens = get_tokens(tokenizer, input_text)
    n_tokens = get_ntokens(tokens)

    # pax({
    #     "input_text" : input_text,
    #     "tokens" : tp(tokens),
    #     "n_tokens" : n_tokens,
    # })

    for i in tqdm(range(n_tokens, n_tokens + 10), "gen tokens"):

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        
        model.eval()
        with torch.no_grad():
            logits = model(tokens, position_ids, attention_mask)

        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        tokens[0, i] = next_token

        # pax({
        #     "tokens" : tp(tokens),
        #     "logits" : tp(logits),
        #     "next_token" : next_token.item(),
        #     "n_tokens" : get_ntokens(tokens),
        # })

    output_text = get_text(tokenizer, tokens[0])

    pax({
        "input_text" : input_text,
        "output_text" : output_text,
    })

    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_output_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_output_len]
        # cut to eos tok if any
        if self.tokenizer.eos_id in toks:
            eos_idx = toks.index(self.tokenizer.eos_id)
            toks = toks[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)
    return (out_tokens, out_logprobs if logprobs else None)

if __name__ == "__main__":

    # {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}

    # Initalize Megatron & JIT.
    initialize_megatron()
    set_jit_fusion_options()

    args = get_args()
    # pax({"args": args})

    # Model, optimizer, and learning rate.
    # model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
    # models, _, _ = setup_model_and_optimizer(
    #     model_provider, ModelType.encoder_or_decoder)
    # model = models[0]

    model = get_model()
    config = get_model_config(model)
    # pax({"model": model, "config" : config})

    tokenizer = get_tokenizer()
    input_text = "lawrence is the fastest cyclist since "
    max_output_len = 64

    output_text = generate(model, tokenizer, input_text, max_output_len)

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
    


