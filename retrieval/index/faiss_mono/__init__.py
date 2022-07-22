# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss

from lutil import pax

from lawrence.index import Index

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class FaissMonoIndex(Index):

    def __init__(self, args, d, index_str):
        super().__init__(args, d)
        self.index = faiss.index_factory(self.din(), index_str)
        
        # pax({
        #     "index" : self.index,
        #     "index / m" : self.index.m,
        #     "ivf" : faiss.extract_index_ivf(self.index),
        # })
    # def init(self):
    #     self.index = faiss.index_factory(self.nfeats, self.index_str)
    #     self.verbose(1)

    def load(self, path):
        self.index = faiss.read_index(path)
        # self.verbose(1)
    
    def verbose(self, v):
        self.c_verbose(self.index, v)

    def to_gpu(self):

        # ~~~~~~~~ move ~~~~~~~~
        index_ivf = faiss.extract_index_ivf(self.index)
        clustering_index = \
            faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index

        # ~~~~~~~~ debug ~~~~~~~~
        # pax({"index": index})

        # ~~~~~~~~ return ~~~~~~~~
        # return index

    # def train(self, data):
    #     self.index.train(data)
    def train(self, data, dirname, timer):

        path = os.path.join(dirname, "empty.faissindex")
        if os.path.isfile(path):
            raise Exception("already trained.")
            return

        timer.push("train")
        self.index.train(data)
        timer.pop()

        timer.push("save")
        faiss.write_index(self.index, path)
        timer.pop()

    def add(self, data):
        self.index.add(data)

    # def save(self, path):
    #     faiss.write_index(self.index, path)

# eof
