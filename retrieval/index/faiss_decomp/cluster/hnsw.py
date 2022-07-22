# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import os

from lutil import pax

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

    # def _forward(
    #         self,
    #         input_data_paths,
    #         _, # centroid_data_paths,
    #         dir_path,
    #         timer,
    #         task,
    # ):

    #     empty_index_path = self.get_empty_index_path(dir_path)

    #     all_output_data_paths, missing_output_data_path_map = \
    #         self.get_missing_output_data_path_map(input_data_paths,dir_path,task)

    #     # pax({
    #     #     "input_data_paths" : input_data_paths,
    #     #     "all_output_data_paths" : all_output_data_paths,
    #     #     "missing_output_data_path_map" : missing_output_data_path_map,
    #     # })

    #     if not missing_output_data_path_map:
    #         return all_output_data_paths

    #     timer.push("load-data")
    #     inp = utils.load_data(input_data_paths, timer)["data"]
    #     timer.pop()

    #     timer.push("init")
    #     hnsw = faiss.read_index(empty_index_path)
    #     self.c_verbose(hnsw, True)
    #     timer.pop()

    #     timer.push("forward-batches")
    #     for batch_index, ((i0, i1), output_data_path) in \
    #         enumerate(missing_output_data_path_map.items()):

    #         sub_inp = inp[i0:i1]
    #         ntrain = self.args.ntrain
    #         print("foward batch [%d:%d] / %d." % (i0, i1, ntrain), flush = True)

    #         # pax({"sub_inp": sub_inp})

    #         timer.push("forward-batch")

    #         timer.push("search")
    #         sub_dists, sub_centroid_ids = hnsw.search(sub_inp, 1)
    #         timer.pop()

    #         # pax({"sub_centroid_ids": sub_centroid_ids})

    #         timer.push("save-data")
    #         utils.save_data({
    #             # "data" : sub_inp,
    #             "centroid_ids" : sub_centroid_ids,
    #         }, output_data_path)
    #         timer.pop()

    #         timer.pop()

    #     timer.pop()

    #     # pax({ ... })

    #     return all_output_data_paths
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

            print("foward batch %d / %d. [ %d vecs ]" % (
                output_index,
                len(missing_output_data_path_map),
                len(inp),
            ), flush = True)

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

        timer.push("train")
        self._train(*args)
        timer.pop()

        timer.push("forward")
        output_data_paths = self._forward(*args, "train")
        timer.pop()

        # pax({"output_data_paths": output_data_paths})

        return output_data_paths

    def add(
            self,
            input_data_paths,
            dir_path,
            timer,
    ):

        timer.push("forward")
        output_data_paths = self._forward(
            input_data_paths,
            None,
            dir_path,
            timer,
            "add",
        )
        timer.pop()

        # pax({"output_data_paths": output_data_paths})

        return output_data_paths
    # def add(self, input_data_paths, dir_path, timer):

    #     raise Exception("call _forward().")

    #     empty_index_path = self.get_empty_index_path(dir_path)
    #     output_data_path = self.get_output_data_path(dir_path, "add")

    #     # pax({
    #     #     "empty_index_path" : empty_index_path,
    #     #     "input_data_path" : input_data_path,
    #     #     "output_data_path" : output_data_path,
    #     # })

    #     if not os.path.isfile(output_data_path):

    #         timer.push("add")

    #         timer.push("load-data")
    #         inp = utils.load_data(input_data_path)["data"]
    #         timer.pop()

    #         timer.push("init")
    #         hnsw = faiss.read_index(empty_index_path)
    #         self.c_verbose(hnsw, True)
    #         # pax({"hnsw": hnsw})
    #         timer.pop()

    #         timer.push("add")
    #         dists, centroid_ids = hnsw.search(inp, 1)
    #         # pax({"dists": dists, "centroid_ids": centroid_ids})
    #         timer.pop()

    #         timer.push("save-data")
    #         utils.save_data({
    #             "data" : inp,
    #             # "centroids" : centroids,
    #             "centroid_ids" : centroid_ids,
    #         }, output_data_path)
    #         timer.pop()

    #         timer.pop()

    #     # if not os.path.isfile(output_data_path):
    #     #     ...

    #     return output_data_path

# eof
