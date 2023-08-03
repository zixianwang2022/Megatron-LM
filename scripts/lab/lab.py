# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import abc
import torch

from megatron import get_args
from scripts import pax


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
        # tokens = torch.cat([
        #     tokens,
        #     torch.full(
        #         (args.seq_length - tokens.numel(),),
        #         self.pad_id, # self.tokenizer.eod,
        #         dtype=torch.long,
        #         device=torch.cuda.current_device(),
        #     ),
        # ], dim=0)
        # tokens = tokens.reshape((1, -1)) # (args.micro_batch_size, -1))

        # pax({
        #     "text" : text,
        #     "tokens" : tokens,
        #     "n_tokens" : self.get_ntokens(tokens),
        # })

        return tokens

    @abc.abstractmethod
    def _detokenize(self, tokens):
        pass

    def detokenize(self, tokens):

        args = get_args()

        # assert tokens.shape == (args.seq_length,)

        n_tokens = self.get_ntokens(tokens)

        text = self._detokenize(tokens[:n_tokens].tolist())

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(text)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # pax({
        #     "n_tokens" : n_tokens,
        #     "tokens" : tokens,
        #     "text" : text,
        # })

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
