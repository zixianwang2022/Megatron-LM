# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss

from lutil import pax, print_rank, print_seq

from retrieval import utils
from retrieval.index import Index
from retrieval.utils.get_index_paths import get_index_str

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class FaissMonoIndex(Index):

#     def __init__(self, args, d, index_str):
#         super().__init__(args, d)
#         self.index = faiss.index_factory(self.din(), index_str)
        
#         # pax({
#         #     "index" : self.index,
#         #     "index / m" : self.index.m,
#         #     "ivf" : faiss.extract_index_ivf(self.index),
#         # })
#     # def init(self):
#     #     self.index = faiss.index_factory(self.nfeats, self.index_str)
#     #     self.verbose(1)

#     def load(self, path):
#         self.index = faiss.read_index(path)
#         # self.verbose(1)
    
#     def verbose(self, v):
#         self.c_verbose(self.index, v)

#     def to_gpu(self):

#         # ~~~~~~~~ move ~~~~~~~~
#         index_ivf = faiss.extract_index_ivf(self.index)
#         clustering_index = \
#             faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
#         index_ivf.clustering_index = clustering_index

#         # ~~~~~~~~ debug ~~~~~~~~
#         # pax({"index": index})

#         # ~~~~~~~~ return ~~~~~~~~
#         # return index

#     # def train(self, data):
#     #     self.index.train(data)
#     def train(self, data, dirname, timer):

#         path = os.path.join(dirname, "empty.faissindex")
#         if os.path.isfile(path):
#             raise Exception("already trained.")
#             return

#         timer.push("train")
#         self.index.train(data)
#         timer.pop()

#         timer.push("save")
#         faiss.write_index(self.index, path)
#         timer.pop()

#     def add(self, data):
#         self.index.add(data)

#     # def save(self, path):
#     #     faiss.write_index(self.index, path)
class FaissMonoIndex(Index):

    # def load(self, path):
    #     self.index = faiss.read_index(path)
    #     # self.verbose(1)
    
    # def verbose(self, v):
    #     self.c_verbose(self.index, v)

    # def to_gpu(self):

    #     # ~~~~~~~~ move ~~~~~~~~
    #     index_ivf = faiss.extract_index_ivf(self.index)
    #     clustering_index = \
    #         faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
    #     index_ivf.clustering_index = clustering_index

    #     # ~~~~~~~~ debug ~~~~~~~~
    #     # pax({"index": index})

    #     # ~~~~~~~~ return ~~~~~~~~
    #     # return index

    def train(self, input_data_paths, dir_path, timer):

        index_str = get_index_str(self.args)
        empty_index_path = self.get_empty_index_path(dir_path)

        # print_seq(index_str)

        timer.push("load-data")
        inp = utils.load_data(input_data_paths, timer)["data"]
        timer.pop()

        # pax({
        #     "inp / shape" : str(inp.shape),
        #     "full_index_path" : full_index_path,
        # })

        timer.push("init")
        index = faiss.index_factory(self.args.nfeats, index_str)
        # pax(0, {"index": index})
        timer.pop()

        # >>>
        self.c_verbose(index, True)
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = \
            faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index
        # <<<

        timer.push("train")
        index.train(inp)
        timer.pop()

        timer.push("save")
        faiss.write_index(index, empty_index_path)
        timer.pop()

    def add(self, data):
        raise Exception("hi.")
        self.index.add(data)

    # def save(self, path):
    #     faiss.write_index(self.index, path)

# eof
