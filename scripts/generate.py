# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from tqdm import tqdm
# from typing import List, Literal, Optional, Tuple, TypedDict
from typing import List, Optional, Tuple

from llama.generation import sample_top_p

from megatron import get_args
from megatron.initialize import initialize_megatron, set_jit_fusion_options

from lab import Lab, LlamaLab, MegatronLab

from lutil import pax


# def debug_preprocess(lab):

#     input_text = "i am the kind of text that you want to tokenize for all your needs. thank you, it's time that you learn how to debug a llm. they are mean and far from lean, and you're a machine."
#     input_tokens = lab.tokenize(input_text)
#     input_ntokens = lab.get_ntokens(input_tokens)

#     word_emb = lab.forward_word_emb(input_tokens)
#     pos_emb = lab.forward_pos_emb(word_emb)

#     # >>>
#     print(lab.model)
#     pax({
#         "input_tokens" : input_tokens,
#         "input_ntokens" : input_ntokens,
#         "word_emb" : word_emb,
#         "pos_emb" : pos_emb,
#     })
#     # <<<

def debug(lab):

    # >>>
    # input_text = "i am the kind of text that you want to tokenize for all your needs. thank you, it's time that you learn how to debug a llm. they are mean and far from lean, and you're a machine."
    input_text = "lawrence is the fastest cyclist since "
    input_ids = lab.tokenize(input_text)
    # input_ntokens = lab.get_ntokens(input_tokens)
    # torch.cuda.manual_seed(0)
    # input_ids = torch.randint(low=0,
    #                           high=args.padded_vocab_size,
    #                           size=(args.seq_length,),
    #                           dtype=torch.long,
    #                           device="cuda")
    # pax({"input_ids": input_ids, "n_tokens": lab.get_ntokens(input_ids)})

    acts = lab.forward_debug(input_ids, debug_layer_idx=None) # input_tokens)

    pax(acts)
    # <<<

    # debug_preprocess(lab)
    # debug_layer(lab)
    # debug_postprocess(lab)

@torch.inference_mode()
def generate(
        lab: Lab,
        input_text: str,
        max_output_len: int,
        # temperature: float = 0.6,
        # temperature: float = 0.,
        temperature: float = 0.99,
        top_p: float = 0.9,
        # logprobs: bool = False,
        # echo: bool = False,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:

    tokens = lab.tokenize(input_text)
    n_tokens = lab.get_ntokens(tokens)

    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(tokens[:n_tokens].tolist())
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # pax({
    #     "input_text" : input_text,
    #     "tokens" : tp(tokens),
    #     "n_tokens" : n_tokens,
    # })
    
    # tokens = tokens.reshape((1, -1))
    for token_idx in tqdm(range(n_tokens, n_tokens + max_output_len),
                          "gen tokens"):

        logits = lab.forward(tokens)

        if temperature > 0:
            # probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            probs = torch.softmax(logits[-1] / temperature, dim=-1)
            # >>>
            
            # <<<
            try:
                next_token = sample_top_p(probs, top_p)
            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~")
                print(lab.detokenize(tokens))
                print("~~~~~~~~~~~~~~~~~~~~~~")
                pax({"token_idx": token_idx, "probs": probs, "e": e})
        else:
            # next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = torch.argmax(logits[-1], dim=-1)

        next_token = next_token.reshape(-1)
        # tokens[token_idx] = next_token
        tokens = torch.cat([tokens, next_token])
        # pax({
        #     "tokens" : tokens,
        #     "tokens'" : torch.cat([tokens, next_token]), # .reshape(1, -1)]),
        # })

        # pax({
        #     "tokens" : tp(tokens),
        #     "logits" : tp(logits),
        #     "next_token" : next_token.item(),
        #     "n_tokens" : lab.get_ntokens(tokens),
        # })
    # tokens = tokens.reshape(-1)

    output_text = lab.detokenize(tokens)

    args = get_args()
    divider = ("~" * 10) + f" [ {args.gen_model} ] " + ("~" * 10)
    print(divider)
    print(output_text)
    print("~" * len(divider))
    pax({
        "input_text" : input_text,
        "output_text" : output_text,
    })

def add_textgen_args(parser):
    group = parser.add_argument_group(title="Text generation.")
    group.add_argument("--gen-model", choices=["megatron","llama"], required=True)
    group.add_argument("--load-llama", required=True)
    return parser

if __name__ == "__main__":

    # {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}

    # Initalize.
    initialize_megatron(extra_args_provider=add_textgen_args)
    set_jit_fusion_options()

    # >>> [llama]
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # <<<

    # Model, tokenizer.
    args = get_args()
    if args.gen_model == "megatron":
        lab = MegatronLab()
    elif args.gen_model == "llama":
        lab = LlamaLab()
    else:
        raise Exception("specialize for '%s'." % args.gen_model)

    # >>>
    # debug(lab)
    # raise Exception("hi.")
    # <<<

    # input_text = "lawrence is the fastest cyclist since "
    # input_text = "the three most important inventions are "
    # input_text = "the most important thing nvidia did was "
    # input_text = "it just makes me so angry that "
    # input_text = "the funniest knock knock joke i ever heard was "
    input_text = "the craziest thing i've ever heard was "
    # input_text = "i'm not the kind of person to " # 300, 0.8
    # input_text = "the best year in history was "
    # input_text = "the best year in history was 1984 because "
    # input_text = "world war 3 will be caused by "

    # Generate.
    output_text = generate(
        lab,
        input_text,
        max_output_len=300,
        temperature=0.8,
    )

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
    


