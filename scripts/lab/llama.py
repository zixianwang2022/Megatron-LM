# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from llama.generation import Llama

from megatron import get_args
from scripts import pax

from .lab import Lab


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
        tokens = tokens[:n_tokens]
        tokens = tokens.reshape((1, -1))

        logits = self.model.forward(tokens, 0)
        logits = logits[0]

        # pax({
        #     "tokens" : tokens,
        #     "logits" : logits,
        # })

        return logits

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def forward_debug_preprocess(self, input_ids, start_pos=0):

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
                            start_pos=0):

        layer = self.model.layers[0]

        # pax({
        #     "layer" : layer,
        #     "hidden_states" : hidden_states,
        #     "attn_mask" : attn_mask,
        #     "freqs_cis" : freqs_cis,
        # })

        acts = {}
        acts["attn_norm"] = layer.attention_norm(hidden_states)
        acts["attn_output"] = layer.attention.forward(
            acts["attn_norm"], start_pos, freqs_cis, attn_mask)
        # acts["attn_residual"] = hidden_states + acts["attn_output"]

        # acts["output"] = layer(x=hidden_states,
        #                        start_pos=0,
        #                        freqs_cis=freqs_cis,
        #                        mask=attn_mask)

        pax({
            "hidden_states" : hidden_states,
            "attn_mask" : attn_mask,
            "freqs_cis" : freqs_cis,
            "--" : "--",
            **acts,
        })

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

        # assert input_ids.shape == (args.seq_length,)
        assert len(input_ids.shape) == 1
        input_ids = input_ids.reshape((1, -1))

        acts = self.forward_debug_model(input_ids)

        pax(acts)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
