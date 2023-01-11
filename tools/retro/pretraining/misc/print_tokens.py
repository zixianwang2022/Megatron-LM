# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from tools.retro.utils import get_gpt_tokenizer


gpt_tokenizer = None

def print_tokens(key, token_ids):

    global gpt_tokenizer
    if gpt_tokenizer is None:
        gpt_tokenizer = get_gpt_tokenizer()

    tokens = gpt_tokenizer.detokenize(token_ids)
    print("%s : %s" % (key, "\\n".join(tokens.splitlines())))
