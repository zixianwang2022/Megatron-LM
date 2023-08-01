# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import abc
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple, TypedDict

# import sys
# sys.path.append("/lustre/fs1/portfolios/adlr/users/lmcafee/llama/2/llama")
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
from megatron.utils import get_ltor_masks_and_position_ids # , unwrap_model
from pretrain_gpt import model_provider

# >>>
from lutil import pax as _pax, tp
def pax(a):
    args = get_args()
    return _pax({
        "gen_model" : args.gen_model,
        "~~" : "~~",
        **{k:tp(v) for k,v in a.items()},
    })
# <<<

# tokenizer_path = "/lustre/fs1/portfolios/adlr/users/lmcafee/llama/2/llama/tokenizer.model"

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

    # @abc.abstractmethod
    # def forward_preprocess(self, inp):
    #     pass
    # @abc.abstractmethod
    # def forward_layer(self, inp):
    #     pass
    # def get_word_emb(self, tokens):
    #     return self.forward_word_emb(tokens)
    @abc.abstractmethod
    def forward_debug(self, tokens):
        pass

class MegatronLab(Lab):

    @classmethod
    def get_model(cls, no_wd_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
        """Setup model and optimizer."""
        args = get_args()

        models = _get_model(model_provider, ModelType.encoder_or_decoder)
        # unwrapped_model = unwrap_model(models, (torchDDP, LocalDDP, Float16Module))
        args.iteration = load_checkpoint(models, None, None)

        # pax({
        #     "models" : models,
        #     "models / 0" : models[0].module.module.language_model.embedding.word_embeddings.weight,
        # })

        return models[0]

    def __init__(self):

        args = get_args()

        if 0:
            tokenizer = get_tokenizer()
            super().__init__(self.get_model(), tokenizer, tokenizer.eod)
        else:
            tokenizer = OriginalLlamaTokenizer(model_path=args.tokenizer_model)
            # pax({
            #     "tokenizer" : tokenizer,
            #     "n_words" : tokenizer.n_words,
            #     # "eod_id" : tokenizer.eod_id,
            #     "pad_id" : tokenizer.pad_id,
            # })
            super().__init__(self.get_model(), tokenizer, tokenizer.pad_id)

        # self.config = get_model_config(self.model)
        # self.seq_length = self.model.config.seq_length

    # def _tokenize(self, text):
    #     return self.tokenizer.tokenize(text, bos=True, eos=False)
    # def _detokenize(self, tokens):
    #     return self.tokenizer.detokenize(tokens)

    def _tokenize(self, text):
        return self.tokenizer.encode(text, bos=True, eos=False)
    def _detokenize(self, tokens):
        return self.tokenizer.decode(tokens)

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

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def forward_debug_preprocess(self, input_ids, position_ids):

        acts = {}
        lm = self.model.module.module.language_model
        acts["hidden_states"] = lm.embedding(input_ids, position_ids, # [s, b, h]
                                             tokentype_ids=None)
        acts["rope_freqs"] = lm.rotary_pos_emb(args.seq_length)

        # pax({
        #     "input_ids" : input_ids,
        #     "weight" : lm.embedding.word_embeddings.weight,
        #     **acts,
        # })

        return acts

    def forward_debug_layer(self, hidden_states, attn_mask, rope_freqs):

        layer = self.model.module.module.language_model.encoder.layers[0]

        # pax({"layer": layer})

        acts = {}
        # acts["attn_norm"] = layer.input_norm(hidden_states)
        # acts["attn_output"], acts["attn_bias"] = \
        #     layer.self_attention(acts["attn_norm"],
        #                          attn_mask,
        #                          rotary_pos_emb=rope_freqs)
        # acts["mlp_norm"] = layer.post_attention_norm(
        acts["output"] = layer(hidden_states=hidden_states,
                               attention_mask=attn_mask,
                               rotary_pos_emb=rope_freqs)

        pax({
            "hidden_states" : hidden_states,
            "attn_mask" : attn_mask,
            "rope_freqs" : rope_freqs,
            **acts,
        })

        return acts

    def forward_debug_model(self, input_ids, position_ids, attention_mask):
        acts = {}
        acts["preprocess"] = self.forward_debug_preprocess(input_ids,position_ids)
        acts["layer"] = self.forward_debug_layer(
            acts["preprocess"]["hidden_states"],
            attention_mask,
            acts["preprocess"]["rope_freqs"])
        pax(acts)

    # def forward_debug(self, tokens):
    def forward_debug(self, input_ids):

        args = get_args()

        assert input_ids.shape == (args.seq_length,)
        input_ids = input_ids.reshape((1, -1))

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            input_ids,
            self.pad_id, # self.tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        
        # pax({
        #     "padded_vocab_size" : args.padded_vocab_size,
        #     "input_ids" : input_ids,
        #     "attention_mask" : attention_mask,
        #     "loss_mask" : loss_mask,
        #     "position_ids" : position_ids,
        # })

        self.model.eval()
        with torch.no_grad():
            activation_map = self.forward_debug_model(input_ids,
                                                      position_ids,
                                                      attention_mask)
            pax({"activation_map": activation_map})

        logits = logits[0]

        pax({
            "tokens" : tp(tokens),
            "logits" : tp(logits),
        })

        return logits
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class LlamaLab(Lab):

    def __init__(self):

        args = get_args()
        # args.seq_length = 128

        generator = Llama.build(
            ckpt_dir=args.load_llama,
            tokenizer_path=args.tokenizer_model, # path,
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

        # pax({
        #     "tokens" : tp(tokens),
        #     "logits" : tp(logits),
        # })

        return logits

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def forward_debug_preprocess(self, input_ids, start_pos=0):

        acts = {}

        acts["hidden_states"] = self.model.tok_embeddings(input_ids)
        freqs_cis = self.model.freqs_cis.to(input_ids.device)
        freqs_cis = freqs_cis[start_pos : start_pos + args.seq_length]
        acts["freqs_cis"] = freqs_cis

        # _bsz, seqlen = input_ids.shape
        mask = torch.full((1, 1, args.seq_length, args.seq_length),
                          float("-inf"),
                          device=input_ids.device)
        mask = torch.triu(mask, diagonal=start_pos+1).type_as(acts["hidden_states"])
        acts["attn_mask"] = mask

        # pax({
        #     "input_ids" : input_ids,
        #     "tok_embeddings" : self.model.tok_embeddings.weight,
        #     **acts,
        # })

        return acts

    def forward_debug_layer(self, hidden_states, attn_mask, freqs_cis):

        layer = self.model.layers[0]

        # pax({"layer": layer})

        acts = {}
        # acts["attn_norm"] = layer.input_norm(hidden_states)
        # acts["attn_output"], acts["attn_bias"] = \
        #     layer.self_attention(acts["attn_norm"],
        #                          attn_mask,
        #                          rotary_pos_emb=rope_freqs)
        # acts["mlp_norm"] = layer.post_attention_norm(
        acts["output"] = layer(x=hidden_states,
                               start_pos=0,
                               freqs_cis=freqs_cis,
                               mask=attn_mask)

        # pax({
        #     "hidden_states" : hidden_states,
        #     "attn_mask" : attn_mask,
        #     "rope_freqs" : rope_freqs,
        #     "acts" : acts,
        # })
        # pax(acts)

        return acts

    def forward_debug_model(self, input_ids):

        # args = get_args()

        acts = {}
        acts["preprocess"] = self.forward_debug_preprocess(input_ids)
        acts["layer"] = self.forward_debug_layer(
            acts["preprocess"]["hidden_states"],
            acts["preprocess"]["attn_mask"],
            acts["preprocess"]["freqs_cis"])

        pax(acts)

    # def forward_debug(self, tokens):
    def forward_debug(self, input_ids):

        args = get_args()

        assert input_ids.shape == (args.seq_length,)
        input_ids = input_ids.reshape((1, -1))

        acts = self.forward_debug_model(input_ids)

        pax(acts)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
    # input_ids = lab.tokenize(input_text)
    # input_ntokens = lab.get_ntokens(input_tokens)
    torch.cuda.manual_seed(0)
    input_ids = torch.randint(low=0,
                              high=args.padded_vocab_size,
                              size=(args.seq_length,),
                              dtype=torch.long,
                              device="cuda")
    # pax({"input_ids": input_ids, "n_tokens": lab.get_ntokens(input_ids)})

    activation_map = lab.forward_debug(input_ids) # input_tokens)

    pax({"activation_map": activation_map})
    # <<<

    debug_preprocess(lab)
    debug_layer(lab)
    debug_postprocess(lab)

    pax({"lab": lab})

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
    for i in tqdm(range(n_tokens, n_tokens + max_output_len), "gen tokens"):

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
    group.add_argument("--load-llama", required=True)
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

    # >>>
    debug(lab)
    raise Exception("hi.")
    # <<<

    # input_text = "lawrence is the fastest cyclist since "
    # input_text = "the three most important inventions are "
    # input_text = "the most important thing nvidia did was "
    # input_text = "it just makes me so angry that "
    # input_text = "the funniest knock knock joke i ever heard was "
    # input_text = "the craziest thing i've ever heard was "
    input_text = "i'm not the kind of person to " # 300, 0.8
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
    


