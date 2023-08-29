# lawrence mcafee

def _add_llama_args(parser):
    group = parser.add_argument_group(title='llama')
    group.add_argument("--_model_family", choices=["megatron", "llama", "hf"])
    group.add_argument("--_model_type", choices=["text", "chat"])
    group.add_argument("--_model_size", choices=["7b", "13b", "70b"])
    return parser

# eof
