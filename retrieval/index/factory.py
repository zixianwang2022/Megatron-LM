# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
# import faiss

# from lutil import pax

from .distrib import DistribIndex
from .faiss_decomp import FaissDecompIndex
from .faiss_mono import FaissMonoIndex
from .faiss_par_add import FaissParallelAddIndex

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class IndexFactory:

    @classmethod
    def get_index_ty(cls, index_ty):
        return {
            "faiss-mono" : FaissMonoIndex,
            "faiss-decomp" : FaissDecompIndex,
            "faiss-par-add" : FaissParallelAddIndex,
            "distrib" : DistribIndex,
        }[index_ty]

    @classmethod
    # def init_index(index_str, nfeats):
    # def init_index(index_ty, index_str, nfeats):
    # def get_index(cls, index_ty, index_str, nfeats):
    # def get_index(cls, args, index_ty, index_str, nfeats):
    # def get_index(cls, args, d, index_ty, index_str, timer):
    #     index_ty = cls.get_index_ty(index_ty)
    #     index = index_ty(args, d, index_str, timer)
    #     return index
    # def get_index(cls, args, timer):
    #     index_ty = cls.get_index_ty(args.index_ty)
    #     index = index_ty(args, timer)
    #     return index
    def get_index(cls, args):
        index_ty = cls.get_index_ty(args.index_ty)
        index = index_ty(args)
        return index

# eof
