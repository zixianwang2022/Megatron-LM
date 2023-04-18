# lawrence mcafee

def init_megatron(cls, workdir):
    '''Custom initialization of Megatron.'''

    # Load args.
    args_path = get_args_path(workdir)
    assert os.path.exists(args_path), "args.json not found in workdir."
    with open(args_path) as f:
        cls.args = types.SimpleNamespace(**json.load(f))
        cls.args.retro_workdir = workdir # just in case workdir moved
        cls.args.rank = 0 # override env
        cls.args.world_size = 1 # override env

    set_global_variables(cls.args)
    set_retro_args(cls.args)
    _initialize_distributed()
    _set_random_seed(cls.args.seed, cls.args.data_parallel_random_init)

@classmethod
def init(cls, workdir):
    '''Initialize Megatron, tokenizers, and datasets.'''

    # Load args.
    cls.init_megatron(workdir)

    cls.tokenizers = types.SimpleNamespace(
        gpt=get_gpt_tokenizer(),
        bert=get_bert_tokenizer(),
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args.add_argument("chunk_id", required=True, type=int)
    args = parser.parse_args()

    pax({"args": args})

