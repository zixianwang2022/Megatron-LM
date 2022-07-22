# lawrence mcafee

? ? ?

# ~~~~~~~~ import ~~~~~~~~
import faiss
import math
import numpy as np
import os

from lutil import pax

from retrieval import utils
from retrieval.index import Index

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class PQStage(Index):
class PQIndex(Index):

    def __init__(self, args, d, m):
        super().__init__(args, d)
        self.m = m
        self.nbits = 8
        # self.pq = faiss.IndexPQ(self.din(), self.m, 8)

    def dout(self):
        return self.m

    def repeat_data(self, data):

        # pax({"data": data})

        # >>>
        # replicate data to 2 * (2**self.nbits) ... hack city.
        ncenters = 2**self.nbits
        # ideal_ntrain = 1 * ncenters # 1, *2
        ideal_ntrain = max(
            int(math.ceil(self.args.ntrain / self.args.nlist)) \
            if self.args.profile_single_encoder else \
            ncenters,
            ncenters,
        )
        # pax({"ideal_ntrain": ideal_ntrain})
        assert len(data) > 0
        # if len(data) < ncenters or self.args.profile_single_encoder:
        if len(data) < ideal_ntrain or self.args.profile_single_encoder:
            repeats = int(math.ceil(ideal_ntrain / len(data)))
            # data = np.repeat(np, repeats, axis = 0)
            data = np.concatenate([ data ] * repeats, axis = 0)
            data = data[:ideal_ntrain]
            # pax({
            #     "data" : data,
            #     "ncenters" : ncenters,
            #     "ideal_ntrain" : ideal_ntrain,
            #     "repeats" : repeats,
            # })
        # <<<

        # pax({"data": str(data.shape)})

        return data

    def train(self, input_data_paths, dir_path, timer):

        empty_index_path = self.get_empty_index_path(dir_path)
        # output_data_path = self.get_output_data_path(dir_path)

        # pax({
        #     "empty_index_path" : empty_index_path,
        #     "input_data_path" : input_data_path,
        #     "output_data_path" : output_data_path,
        # })

        if not os.path.isfile(empty_index_path):

            timer.push("train")

            timer.push("load-data")
            inp = utils.load_data(input_data_paths, timer)["data"]
            timer.pop()

            timer.push("repeat-data")
            inp = self.repeat_data(inp)
            timer.pop()

            timer.push("init")
            pq = faiss.IndexPQ(self.din(), self.m, self.nbits)
            self.c_verbose(pq, True)
            timer.pop()

            timer.push("train")
            pq.train(inp)
            timer.pop()

            timer.push("save")
            faiss.write_index(pq, empty_index_path)
            timer.pop()

            timer.pop()

        # if not os.path.isfile(output_data_path):
        #     ...

        return None

    # def add(self, data_map_0):
    #     raise Exception("hi.")
    #     self.pq.add(data_map_0["data"])
    def add(self, input_data_paths, dir_path, timer):

        empty_index_path = self.get_empty_index_path(dir_path)
        full_index_path = self.get_full_index_path(dir_path)

        # pax({
        #     "empty_index_path" : empty_index_path,
        #     "full_index_path" : full_index_path,
        #     "input_data_path" : input_data_path,
        # })

        if not os.path.isfile(full_index_path):

            timer.push("add")

            timer.push("load-data")
            inp = utils.load_data(input_data_paths, timer)["data"]
            timer.pop()

            timer.push("repeat-data")
            inp = self.repeat_data(inp)
            timer.pop()

            # >>>
            # ideal data size, for profiling ... hack city.
            # if self.args.profile_single_encoder:
            #     ideal_ntrain = int(math.ceil(self.args.ntrain / self.args.nlist))
            #     # pax({"ideal_ntrain": ideal_ntrain})
            #     repeats = int(math.ceil(ideal_ntrain / len(inp)))
            #     inp = np.concatenate([ inp ] * repeats, axis = 0)
            #     inp = inp[:ideal_ntrain]
            # <<<

            timer.push("init")
            pq = faiss.read_index(empty_index_path)
            self.c_verbose(pq, True)
            timer.pop()

            timer.push("add")
            pq.add(inp)
            timer.pop()

            timer.push("save")
            faiss.write_index(pq, full_index_path)
            timer.pop()

            timer.pop()

        # if not os.path.isfile(output_data_path):
        #     ...

        return None

# eof
