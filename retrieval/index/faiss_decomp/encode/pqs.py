# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
# import numpy as np
import os

from lutil import pax

from retrieval.index import Index
import retrieval.utils as utils

# from .pq import PQIndex

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class PQsStage(Index):
# class PQsIndex(Index):

#     def __init__(self, args, d, stage_str):
#         super().__init__(args, d)
    
#         # pax({"stage_str": stage_str})

#         assert stage_str.startswith("PQ")
#         self.m = int(stage_str.replace("PQ", ""))

#         # pax({"args": self.args})

#         # if args.profile_single_encoder:
#         #     self.pqs = [ PQIndex(args, d, self.m) ]
#         # else:
#         #     raise Exception("hi.")
#         #     self.pqs = [ PQIndex(args, d, self.m) for _ in range(args.nlist) ]

#     def dout(self):
#         return self.m

#     # def verbose(self, v):
#     #     [ pq.verbose(v) for pq in self.pqs ]

#     @classmethod
#     def save_data_slice(cls, path, data, centroid_ids, centroid_id, timer):
#         if not os.path.isfile(path):

#             timer.push("get")
#             indexes = np.where(centroid_ids == centroid_id)
#             sub_data = data[indexes[0]]
#             timer.pop()

#             timer.push("save")
#             utils.save_data({"data": sub_data}, path)
#             timer.pop()

#     def train(self, input_data_paths, dir_path, timer):

#         # >>>
#         inp_map = utils.load_data(input_data_paths, timer)
#         pax({"inp": inp})
#         raise Exception("load residuals, not data.")
#         # <<<

#         timer.push("load-data")
#         input_data_map = utils.load_data(input_data_paths, timer)
#         input_data = input_data_map["data"]
#         centroid_ids = input_data_map["centroid_ids"]
#         timer.pop()

#         # pax({
#         #     "input_data_paths" : input_data_paths,
#         #     "input_data" : str(input_data.shape),
#         #     "centroid_ids" : str(centroid_ids.shape),
#         # })

#         # for centroid_id in range(self.args.nlist):
#         for centroid_id, pq in enumerate(self.pqs):

#             # timer.push(str(centroid_id))
#             timer.push("pq")

#             print(">> train pq %d / %d." % (centroid_id, self.args.nlist))
#             # time.sleep(1) # ... adds 4M seconds for full run

#             sub_dir_path = utils.make_sub_dir(dir_path, str(centroid_id))
#             sub_input_data_path = os.path.join(sub_dir_path, "train_input.hdf5")

#             timer.push("slice-data") # 'sub-data'
#             self.save_data_slice(
#                 sub_input_data_path,
#                 input_data,
#                 centroid_ids,
#                 centroid_id,
#                 timer,
#             )
#             timer.pop()

#             timer.push("train")
#             pq.train(
#                 [ sub_input_data_path ],
#                 sub_dir_path,
#                 timer,
#             )
#             timer.pop()

#             timer.pop()

#         # ?

#         return None

#     def add(self, input_data_paths, dir_path, timer):

#         timer.push("load-data")
#         input_data_map = utils.load_data(input_data_paths, timer)
#         input_data = input_data_map["data"]
#         centroid_ids = input_data_map["centroid_ids"]
#         timer.pop()

#         # pax({"input_data_paths": input_data_paths})

#         # for centroid_id in range(self.args.nlist):
#         for centroid_id, pq in enumerate(self.pqs):

#             timer.push("pq") # str(centroid_id)

#             print(">> add pq %d / %d." % (centroid_id, self.args.nlist))
#             # time.sleep(1) # ... adds 4M seconds for full run

#             sub_dir_path = utils.make_sub_dir(dir_path, str(centroid_id))
#             sub_input_data_path = os.path.join(sub_dir_path, "add_input.hdf5")

#             timer.push("slice-data") # 'sub-data'
#             self.save_data_slice(
#                 sub_input_data_path,
#                 input_data,
#                 centroid_ids,
#                 centroid_id,
#                 timer,
#             )
#             timer.pop()

#             timer.push("add")
#             pq.add(
#                 [ sub_input_data_path ],
#                 sub_dir_path,
#                 timer,
#             )
#             timer.pop()

#             timer.pop()

#         # ?

#         return None
class PQsIndex(Index):

    def __init__(self, args, d, stage_str):
        super().__init__(args, d)
    
        assert stage_str.startswith("PQ")
        self.m = int(stage_str.replace("PQ", ""))
        self.nbits = 8

    def dout(self):
        return self.m

    def train(self, input_data_paths, dir_path, timer):

        empty_index_path = self.get_empty_index_path(dir_path)
        # output_data_path = self.get_output_data_path(dir_path)

        if os.path.isfile(empty_index_path):
            return None

        residual_data_paths = [ p["residuals"] for p in input_data_paths ]
        # centroid_id_data_paths = [ p["centroid_ids"] for p in input_data_paths ]

        # pax({
        #     "input_data_paths" : input_data_paths,
        #     "input_data_paths / 0" : input_data_paths[0],
        #     "residual_data_paths" : residual_data_paths,
        #     "centroid_id_data_paths" : centroid_id_data_paths,
        # })

        timer.push("load-data")
        # >>>
        # input_data_map = utils.load_data(input_data_paths, timer)
        # input_data = input_data_map["residuals"]
        # centroid_ids = input_data_map["centroid_ids"]
        # +++
        input_data = utils.load_data(residual_data_paths, timer)["residuals"]
        # centroid_ids = utils.load_data(centroid_id_data_paths, timer)["centroid_ids"]
        # assert len(input_data) == len(centroid_ids)
        # <<<
        timer.pop()

        # pax({
        #     # "input_data_paths" : input_data_paths,
        #     "input_data" : str(input_data.shape),
        #     "centroid_ids" : str(centroid_ids.shape),
        # })

        timer.push("init")
        pq = faiss.IndexPQ(self.din(), self.m, self.nbits)
        self.c_verbose(pq, True)
        timer.pop()

        timer.push("train")
        pq.train(input_data)
        timer.pop()

        timer.push("save")
        faiss.write_index(pq, empty_index_path)
        timer.pop()

    # def add(self, data_map_0):
    #     raise Exception("hi.")
    #     self.pq.add(data_map_0["data"])
    def add(self, input_data_paths, dir_path, timer):

        empty_index_path = self.get_empty_index_path(dir_path)
        full_index_path = self.get_full_index_path(dir_path)

        if os.path.isfile(full_index_path):
            return None

        timer.push("init")
        pq = faiss.read_index(empty_index_path)
        self.c_verbose(pq, True)
        timer.pop()

        # pax({"pq": pq})

        # timer.push("add")

        for input_index, input_data_path_item in enumerate(input_data_paths):

            timer.push("load-data")
            input_data_path = input_data_path_item["residuals"]
            # centroid_id_data_path = input_data_path_item["centroid_ids"]
            input_data = utils.load_data([ input_data_path ], timer)["residuals"]
            # centroid_ids = utils.load_data([ centroid_id_data_path ], timer)["centroid_ids"]
            # assert len(input_data) == len(centroid_ids)
            timer.pop()

            # pax({
            #     "input_data_path" : input_data_path,
            #     "centroid_id_data_path" : centroid_id_data_path,
            #     "input_data" : str(input_data.shape),
            #     "centroid_ids" : str(centroid_ids.shape),
            # })

            print("pqs / add,  batch %d / %d. [ %d vecs ]" % (
                input_index,
                len(input_data_paths),
                len(input_data),
            ), flush = True)

            timer.push("add")
            pq.add(input_data)
            timer.pop()

        timer.push("save")
        faiss.write_index(pq, full_index_path)
        timer.pop()

        # timer.pop()

# eof
