# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from .faiss_base import FaissBaseIndex
from .faiss_decomp import FaissDecompIndex
from .faiss_par_add import FaissParallelAddIndex

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class IndexFactory:

    @classmethod
    def get_index_ty(cls, index_ty):
        return {
            "faiss-base" : FaissBaseIndex,
            "faiss-decomp" : FaissDecompIndex,
            "faiss-par-add" : FaissParallelAddIndex,
        }[index_ty]

    @classmethod
    def get_index(cls, args):
        index_ty = cls.get_index_ty(args.index_ty)
        index = index_ty(args)
        return index

# eof
