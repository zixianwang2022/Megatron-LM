# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import os
import torch

from lutil import pax

# from .mkdir import mkdir
from . import mkdir

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def get_index_paths(args):

#     # ~~~~~~~~ save paths ~~~~~~~~
#     # index_filename_prefix = "%s__t%d_c%d" % (
#     #     args.index_str,
#     #     args.ntrain,
#     #     args.ncluster,
#     # )
#     index_filename_prefix = "%s__t%d" % (
#         args.index_str,
#         args.ntrain,
#     )
#     trained_index_filename = "%s__trained.faissindex" % index_filename_prefix
#     added_index_filename = "%s__added.faissindex" % index_filename_prefix
#     args.trained_index_path = os.path.join(
#         args.base_dir,
#         "index",
#         "%s-%s" % (args.index_ty, args.data_ty),
#         trained_index_filename,
#     )
#     args.added_index_path = os.path.join(
#         args.base_dir,
#         "index",
#         "%s-%s" % (args.index_ty, args.data_ty),
#         added_index_filename,
#     )

#     # pax({"args": args})
# def get_index_dirname(args):

#     # ~~~~~~~~ save paths ~~~~~~~~
#     args.index_dirname = os.path.join(
#         args.base_dir,
#         "index",
#         "%s-%s" % (args.index_ty, args.data_ty),
#         "%s__t%d" % (
#             args.index_str,
#             args.ntrain,
#         ),
#     )

#     mkdir(os.path.dirname(args.index_dirname))
#     mkdir(args.index_dirname)

#     # pax({
#     #     "args": args,
#     #     "dirname" : args.index_dirname,
#     #     "par dirname" : os.path.dirname(args.index_dirname),
#     # })
def get_index_str(args):
    return "OPQ%d_%d,IVF%d_HNSW%d,PQ%d" % (
        args.pq_m,
        args.ivf_dim,
        args.ncluster,
        args.hnsw_m,
        args.pq_m,
    )

def get_index_dirname(args):

    # ~~~~~~~~ save paths ~~~~~~~~
    index_str = get_index_str(args)
    # pax({"args": args, "index_str": index_str})
    index_dirname = os.path.join(
        args.base_dir,
        "index",
        "%s-%s" % (args.index_ty, args.data_ty),
        "%s__t%d__a%d__w%d" % (
            index_str,
            args.ntrain,
            args.nadd,
            torch.distributed.get_world_size(),
        ),
    )

    mkdir(os.path.dirname(index_dirname))
    mkdir(index_dirname)

    # pax({
    #     "args": args,
    #     "dirname" : index_dirname,
    #     "par dirname" : os.path.dirname(index_dirname),
    # })

    return index_dirname

# eof
