# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


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
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
