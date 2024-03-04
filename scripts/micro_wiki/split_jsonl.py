# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import json
import os
from tqdm import tqdm

from megatron.tokenizer.tokenizer import (
    _GPT2BPETokenizer,
    _GPTSentencePieceTokenizer,
)

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_bpe_tokenizer():
    return _GPT2BPETokenizer(
        vocab_file="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/vocab/gpt2-vocab.json",
        merge_file="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/vocab/gpt2-merges.txt",
    )

def get_sentencepiece_tokenizer():
    return _GPTSentencePieceTokenizer("/lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")

if __name__ == "__main__":

    # tokenizer_ty = "bpe"
    # tokenizer_ty = "sentencepiece"

    corpus_dir = "/lustre/fs6/portfolios/adlr/users/lmcafee/corpus-530b"
    # src_path = "/lustre/fs6/portfolios/adlr/users/lmcafee/corpus-530b/wiki-tiny/Wikipedia_en_ftfy_id_shuf.jsonl"
    src_path = os.path.join(corpus_dir, "wiki-tiny/wiki-200k.jsonl")
    entries = open(src_path).read().splitlines()

    tokenizer = get_bpe_tokenizer()
    # tokenizer = get_sentencepiece_tokenizer()

    seq_length = 2048
    n_samples = 5000
    total_tokens = n_samples * seq_length

    pbar = tqdm(entries) # , "entries")
    crnt_tokens = 0
    for entry_idx, entry in enumerate(pbar):
        pbar.set_description("crnt tokens %.0f" % (100 * crnt_tokens / total_tokens))
        entry = json.loads(entry)
        tokens = tokenizer.tokenize(entry["text"])
        crnt_tokens += len(tokens)
        # pax("entry, tokens")
        if crnt_tokens >= total_tokens:
            break

    # sub_entries = entries[:(entry_idx+1)]

    dst_path = os.path.join(corpus_dir, "micro-wiki/micro-wiki-%d.jsonl" % (entry_idx + 1))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        f.write("\n".join(entries[:(entry_idx+1)]))

    pax({"entries": len(entries)}, "seq_length, n_samples, total_tokens, crnt_tokens, entry_idx")

# eof
