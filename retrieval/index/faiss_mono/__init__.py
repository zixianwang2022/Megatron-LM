# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from datetime import timedelta
import faiss
import os
import torch

from lutil import pax, print_rank, print_seq

from retrieval import utils
from retrieval.index import Index
from retrieval.utils.get_index_paths import get_index_str

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def barrier():
    # torch.distributed.monitored_barrier(timeout = timedelta(days = 1))
    torch.distributed.barrier()

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

    def _train(self, input_data_paths, dir_path, timer):

        assert torch.distributed.get_rank() == 0

        index_str = get_index_str(self.args)
        empty_index_path = self.get_empty_index_path(dir_path)

        if os.path.isfile(empty_index_path):
            # raise Exception("empty index already exists.")
            return

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
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = \
            faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index
        self.c_verbose(index, True)
        self.c_verbose(index_ivf, True)
        self.c_verbose(index_ivf.quantizer, True)
        self.c_verbose(index_ivf.clustering_index, True)
        # <<<

        timer.push("train")
        index.train(inp)
        timer.pop()

        timer.push("save")
        faiss.write_index(index, empty_index_path)
        timer.pop()

    def train(self, input_data_paths, dir_path, timer):

        barrier()

        if torch.distributed.get_rank() == 0:
            timer.push("train")
            self._train(input_data_paths, dir_path, timer)
            timer.pop()

        barrier()

    def _add_full_batch(self, input_data_paths, dir_path, timer):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)
        full_index_path = self.get_full_index_path(dir_path)

        if os.path.isfile(full_index_path):
            return

        timer.push("load-data")
        inp = utils.load_data(input_data_paths, timer)["data"]
        timer.pop()

        # pax({
        #     "inp / shape" : str(inp.shape),
        #     "full_index_path" : full_index_path,
        # })

        timer.push("init")
        index = faiss.read_index(empty_index_path)
        # pax(0, {"index": index})
        timer.pop()

        # >>>
        index_ivf = faiss.extract_index_ivf(index)
        # clustering_index = \
        #     faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        # index_ivf.clustering_index = clustering_index
        self.c_verbose(index, True)
        self.c_verbose(index_ivf, True)
        self.c_verbose(index_ivf.quantizer, True)
        # self.c_verbose(index_ivf.clustering_index, True)
        # <<<

        timer.push("add")
        index.add(inp)
        timer.pop()

        timer.push("save")
        faiss.write_index(index, full_index_path)
        timer.pop()

    def _add_mini_batch(self, input_data_paths, dir_path, timer):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)
        full_index_path = self.get_full_index_path(dir_path)

        if os.path.isfile(full_index_path):
            return

        timer.push("init")
        index = faiss.read_index(empty_index_path)
        timer.pop()

        for batch_id, input_data_path in enumerate(input_data_paths):

            print_rank("faiss-mono / add, batch %d / %d." % (
                batch_id,
                len(input_data_paths),
            ))

            timer.push("load-data")
            inp = utils.load_data([ input_data_path ], timer)["data"]
            timer.pop()

            # >>>
            index_ivf = faiss.extract_index_ivf(index)
            # clustering_index = \
            #     faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
            # index_ivf.clustering_index = clustering_index
            self.c_verbose(index, True)
            self.c_verbose(index_ivf, True)
            self.c_verbose(index_ivf.quantizer, True)
            # self.c_verbose(index_ivf.clustering_index, True)
            # <<<

            timer.push("add")
            index.add(inp)
            timer.pop()

        timer.push("save")
        faiss.write_index(index, full_index_path)
        timer.pop()

    def add(self, input_data_paths, dir_path, timer):

        barrier()

        if torch.distributed.get_rank() == 0:
            timer.push("add")
            # self._add_full_batch(input_data_paths, dir_path, timer)
            self._add_mini_batch(input_data_paths, dir_path, timer)
            timer.pop()

        barrier()

# eof
