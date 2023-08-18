# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
# from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

# from llama.generation import Llama

from megatron import get_args
from scripts import pax

from .lab import Lab


class HFLab(Lab):

    def __init__(self, max_seq_len=None, max_batch_size=None):

        # >>>
        # import torch
        # torch.set_default_device()
        # <<<

        args = get_args()
        
        # >>>
        # state_dict = load_state_dict(args.load)
        # pax({"state_dict": state_dict})
        # <<<

        # import torch
        # state_dict = torch.load(args.load)
        # pax({"state_dict": state_dict})

        # import torch
        # torch.set_default_device("cpu")
        # pax({"default device": torch.device}) # get_default_device()})

        model = LlamaForCausalLM.from_pretrained(args.load, device_map="cpu")
        # model.to("cuda")
        tokenizer = LlamaTokenizer.from_pretrained(args.load)

        # >>>
        if torch.distributed.get_rank() == 0:
            print(model)
        pax({
            "device" : str(model.device),
            "nparams" : sum(t.numel() for t in model.parameters()),
            "params / 0" : list(model.parameters())[0],
        })
        # <<<

        super().__init__(
            model,
            tokenizer,
            # tokenizer.bos_token_id,
            # tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )

        # pax({
        #     "args / load": args.load,
        #     "model" : self.model,
        #     "tokenizer" : self.tokenizer,
        # })

    def _tokenize(self, text):
        return self.tokenizer.encode(text)

    def _detokenize(self, tokens):
        return self.tokenizer.decode(tokens)

    def forward(self, tokens):

        args = get_args()

        # assert tokens.shape == (args.seq_length,)

        # n_tokens = self.get_ntokens(tokens)
        # tokens = tokens[:n_tokens]
        tokens = tokens.reshape((1, -1))

        # pax({"model": self.model, "tokens": tokens})

        # logits = self.model.forward(tokens, 0)
        logits = self.model(tokens)
        logits = logits.logits[0]
        # logits = logits[-1]

        # pax({
        #     "tokens" : tokens,
        #     "logits" : logits,
        #     "logits / 0" : logits[0],
        #     "logits / -1" : logits[-1],
        # })

        return logits

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ... downstream task testing ...

    def eval(self):
        raise Exception("hi.")
        pass

    def set_input_tensor(self, tensor):
        raise Exception("hi.")
        assert tensor is None

    def __call__(self, input_ids, position_ids, attention_mask, inference_params):

        raise Exception("hi.")

        # tokens = torch.tensor(
        #     self._tokenize(text),
        #     dtype=torch.long,
        #     device=torch.cuda.current_device())

        # try:
        logits = self.model.forward(input_ids, 0)
        # except Exception as e:
        #     pax({"input_ids": input_ids, "e": e})
        # logits = logits.transpose(0, 1)

        # pax({
        #     "input_ids" : input_ids,
        #     # "position_ids" : position_ids,
        #     # "attention_mask" : attention_mask,
        #     "logits" : logits,
        # })

        return logits

    # @property
    # def eos_token(self):
    #     return self.tokenizer.eos_id
    @property
    def eod(self):
        raise Exception("hi.")
        return self.tokenizer.eos_id

    # def decode(self, token_ids):
    #     return self.tokenizer.decode(token_ids)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def forward_debug_preprocess(self, input_ids, start_pos=0):

        raise Exception("hi.")

        # args = get_args()
        # seq_length = args.seq_length
        seq_length = input_ids.shape[1]

        acts = {}

        acts["hidden_states"] = self.model.tok_embeddings(input_ids)

        freqs_cis = self.model.freqs_cis.to(input_ids.device)
        freqs_cis = freqs_cis[start_pos:(start_pos + seq_length)]
        acts["freqs_cis"] = freqs_cis

        # _bsz, seqlen = input_ids.shape
        mask = torch.full((1, 1, seq_length, seq_length),
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

    def forward_debug_layer(self, hidden_states, attn_mask, freqs_cis,
                            layer_idx, debug, start_pos=0):

        raise Exception("hi.")

        layer = self.model.layers[layer_idx]

        acts = {}
        acts["attn_norm"] = layer.attention_norm(hidden_states)
        acts["attn_output"] = layer.attention.forward(
            acts["attn_norm"], start_pos, freqs_cis, attn_mask,
            debug=False) # debug)
        acts["hidden"] = hidden_states + acts["attn_output"]
        acts["mlp_norm"] = layer.ffn_norm(acts["hidden"])
        acts["mlp_output"] = layer.feed_forward.forward(acts["mlp_norm"])
        acts["output"] = acts["hidden"] + acts["mlp_output"]
        acts["output [gold]"] = layer(x=hidden_states,
                                      start_pos=0,
                                      freqs_cis=freqs_cis,
                                      mask=attn_mask)

        # >>>
        if debug:
            pax({
                "layer_idx" : layer_idx,
                "hidden_states" : hidden_states,
                "attn_mask" : attn_mask,
                "freqs_cis" : freqs_cis,
                "--" : "--",
                # "attn_norm / w" : layer.attention_norm.weight,
                **acts,
            })
        # <<<

        return acts

    def forward_debug_layers(self, hidden_states, attn_mask, freqs_cis,
                             debug_layer_idx, start_pos=0):

        raise Exception("hi.")

        # >>>
        # pax({"attn_norms / w": [ layer.attention_norm.weight
        #                          for layer in self.model.layers ]})
        # <<<

        outs = []
        for layer_idx, layer in enumerate(tqdm(self.model.layers, "layers")):
            # >>>
            # inp = hidden_states
            # <<<
            # hidden_states = layer(x=hidden_states,
            #                       start_pos=0,
            #                       freqs_cis=freqs_cis,
            #                       mask=attn_mask,
            #                       debug=layer_idx==1)
            acts = self.forward_debug_layer(
                hidden_states,
                attn_mask,
                freqs_cis,
                layer_idx,
                layer_idx == debug_layer_idx,
            )
            hidden_states = acts["output [gold]"]
            # >>>
            # out = hidden_states
            # if layer_idx == 1:
            #     pax({
            #         "inp" : inp,
            #         "out" : out,
            #     })
            # <<<
            outs.append(hidden_states)

        acts = {"hidden_states": outs[-1]}

        # pax({
        #     "hidden_states" : hidden_states,
        #     "attn_mask" : attn_mask,
        #     "freqs_cis" : freqs_cis,
        #     "--" : "--",
        #     "outs" : outs,
        # })
        # pax({"outs / -1": outs[-1]})

        return acts

    def forward_debug_postprocess(self, hidden_states):

        raise Exception("hi.")

        acts = {}
        acts["norm"] = self.model.norm(hidden_states)
        acts["output"] = self.model.output(acts["norm"]).float()

        pax({
            "hidden_states" : hidden_states,
            "--" : "--",
            **acts,
        })

        return acts

    def forward_debug_model(self, input_ids, debug_layer_idx):

        raise Exception("hi.")

        # args = get_args()

        acts = {}
        acts["preprocess"] = self.forward_debug_preprocess(input_ids)
        # acts["layer"] = self.forward_debug_layer(
        #     acts["preprocess"]["hidden_states"],
        #     acts["preprocess"]["attn_mask"],
        #     acts["preprocess"]["freqs_cis"])
        acts["layers"] = self.forward_debug_layers(
            acts["preprocess"]["hidden_states"],
            acts["preprocess"]["attn_mask"],
            acts["preprocess"]["freqs_cis"],
            debug_layer_idx)
        acts["postprocess"] = self.forward_debug_postprocess(
            acts["layers"]["hidden_states"])

        # acts["output [gold]"] = self.model(input_ids, 0)
        acts["output [gold]"] = self.forward(input_ids)

        pax(acts)

    # def forward_debug(self, tokens):
    def forward_debug(self, input_ids, debug_layer_idx):

        raise Exception("hi.")

        args = get_args()

        # assert input_ids.shape == (args.seq_length,)
        assert len(input_ids.shape) == 1
        input_ids = input_ids.reshape((1, -1))

        acts = self.forward_debug_model(input_ids, debug_layer_idx)

        pax(acts)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
