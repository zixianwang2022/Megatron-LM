# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import os
import torch

from lutil import pax, print_rank, print_seq

from retrieval.index import Index
import retrieval.utils as utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class HNSWStage(Index):
class HNSWIndex(Index):

    def __init__(self, args, d, m):
        super().__init__(args, d)
        self.m = m
        # self.hnsw = faiss.IndexHNSWFlat(d, m)

    def dout(self):
        return self.din()

    # def verbose(self, v):
    #     self.c_verbose(self.hnsw, v)

    def _train(
            self,
            input_data_paths,
            centroid_data_paths,
            dir_path,
            timer,
    ):

        empty_index_path = self.get_empty_index_path(dir_path)

        if os.path.isfile(empty_index_path):
            return

        timer.push("load-data")
        centroids = utils.load_data(centroid_data_paths, timer)["centroids"]
        timer.pop()

        # pax({"centroids": centroids})

        timer.push("init")
        hnsw = faiss.IndexHNSWFlat(self.din(), self.m)
        self.c_verbose(hnsw, True)
        timer.pop()

        timer.push("train")
        hnsw.train(centroids)
        timer.pop()

        timer.push("add")
        hnsw.add(centroids)
        timer.pop()

        timer.push("save")
        faiss.write_index(hnsw, empty_index_path)
        timer.pop()

    def _forward(
            self,
            input_data_paths,
            _, # centroid_data_paths,
            dir_path,
            timer,
            task,
    ):

        empty_index_path = self.get_empty_index_path(dir_path)

        all_output_data_paths, missing_output_data_path_map = \
            self.get_missing_output_data_path_map(input_data_paths,dir_path,task)

        all_output_data_paths = [ {
            "data" : i,
            "centroid_ids" : o,
        } for i, o in zip(input_data_paths, all_output_data_paths) ]

        # pax({
        #     "input_data_paths" : input_data_paths,
        #     "all_output_data_paths" : all_output_data_paths,
        #     "all_output_data_paths / 0" : all_output_data_paths[0],
        #     "missing_output_data_path_map" : missing_output_data_path_map,
        # })
        # print_seq(list(missing_output_data_path_map.values()))

        if not missing_output_data_path_map:
            return all_output_data_paths

        timer.push("init")
        hnsw = faiss.read_index(empty_index_path)
        self.c_verbose(hnsw, True)
        timer.pop()

        timer.push("forward-batches")
        for output_index, (input_index, output_data_path) in \
            enumerate(missing_output_data_path_map.items()):

            timer.push("load-data")
            input_data_path = input_data_paths[input_index]
            inp = utils.load_data([ input_data_path ], timer)["data"]
            timer.pop()

            print_rank("foward batch %d / %d. [ %d vecs ]" % (
                output_index,
                len(missing_output_data_path_map),
                len(inp),
            )) # , flush = True)

            timer.push("forward-batch")

            timer.push("search")
            dists, centroid_ids = hnsw.search(inp, 1)
            timer.pop()

            # pax({"centroid_ids": centroid_ids})

            timer.push("save-data")
            utils.save_data({
                "centroid_ids" : centroid_ids,
            }, output_data_path)
            timer.pop()

            timer.pop()

        timer.pop()

        # pax({ ... })

        return all_output_data_paths

    def train(self, *args):

        timer = args[-1]

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            timer.push("train")
            self._train(*args)
            timer.pop()

        torch.distributed.barrier()

        timer.push("forward")
        output_data_paths = self._forward(*args, "train")
        timer.pop()

        torch.distributed.barrier()

        # pax({"output_data_paths": output_data_paths})

        return output_data_paths

    def add(
            self,
            input_data_paths,
            dir_path,
            timer,
    ):

        torch.distributed.barrier()

        timer.push("forward")
        output_data_paths = self._forward(
            input_data_paths,
            None,
            dir_path,
            timer,
            "add",
        )
        timer.pop()

        torch.distributed.barrier()

        # pax({"output_data_paths": output_data_paths})
        # print_seq(output_data_paths)

        return output_data_paths

# eof
