# lawrence mcafee

from megatron import get_args

from lutil import pax as _pax, tp


def pax(a):
    try:
        args = get_args()
        return _pax({
            "gen_model" : args.gen_model,
            "~~" : "~~",
            # **{k:tp(v) for k,v in a.items()},
            **a,
        })
    except Exception as e:
        return _pax(a)

# eof
