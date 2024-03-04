# lawrence mcafee

set -u

pip install nltk

cd /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/retro-mcore-data

# python -m scripts.micro_wiki.split_jsonl

MICRO_WIKI_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/corpus-530b/micro-wiki"

# --chunk-size 1000
# python tools/preprocess_data.py \
#   --input "${MICRO_WIKI_DIR}/micro-wiki-14061.jsonl" \
#   --output-prefix "${MICRO_WIKI_DIR}/micro-wiki-bpe" \
#   --tokenizer-type "GPT2BPETokenizer" \
#   --vocab-file "/lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/vocab/gpt2-vocab.json" \
#   --merge-file "/lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/vocab/gpt2-merges.txt" \
#   --workers 40

# --chunk-size 1000
python tools/preprocess_data.py \
  --input "${MICRO_WIKI_DIR}/micro-wiki-14061.jsonl" \
  --output-prefix "${MICRO_WIKI_DIR}/micro-wiki-sentencepiece" \
  --tokenizer-type "GPTSentencePieceTokenizer" \
  --tokenizer-model "/lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model" \
  --workers 40

# eof
