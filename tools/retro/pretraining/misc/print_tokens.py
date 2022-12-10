# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tools.retro.utils import get_gpt_tokenizer


gpt_tokenizer = None

def print_tokens(key, token_ids):

    global gpt_tokenizer
    if gpt_tokenizer is None:
        gpt_tokenizer = get_gpt_tokenizer()

    tokens = gpt_tokenizer.detokenize(token_ids)
    print("%s : %s" % (key, "\\n".join(tokens.splitlines())))
