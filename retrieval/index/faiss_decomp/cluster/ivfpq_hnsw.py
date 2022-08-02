# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from lutil import pax, print_rank, print_seq

from retrieval.index import Index
import retrieval.utils as utils

from .hnsw import HNSWIndex
from .ivfpq import IVFPQIndex

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class IVFHNSWStage(Index):
# class IVFHNSW(Index):
# class IVFHNSWIndex(Index):
class IVFPQHNSWIndex(Index):

    # def __init__(self, args, d, stage_str):
    #     super().__init__(args, d)

    #     tokens = stage_str.split("_")
    #     assert len(tokens) == 2
    #     assert tokens[0].startswith("IVF") # redundant
    #     assert tokens[1].startswith("HNSW")

    #     self.nlist = int(tokens[0].replace("IVF", ""))
    #     self.m = int(tokens[1].replace("HNSW", ""))

    #     args.nlist = self.nlist

    #     self.ivfpq = IVFPQIndex(args, d, self.nlist)
    #     self.hnsw = HNSWIndex(args, d, self.m)
    def __init__(self, args):
        # super().__init__(args, args.ivf_dim, None) # args.pq_dim)
        super().__init__(args)
        self.ivfpq = IVFPQIndex(args)
        self.hnsw = HNSWIndex(args)

    # def dout(self):
    #     return self.din()

    # def verbose(self, v):
    #     self.ivf.verbose(v)
    #     self.hnsw.verbose(v)

    def train(self, input_data_paths, dir_path, timer):

        ivfpq_dir_path = utils.make_sub_dir(dir_path, "ivfpq")
        hnsw_dir_path = utils.make_sub_dir(dir_path, "hnsw")

        timer.push("ivfpq")
        ivfpq_output_data_paths = self.ivfpq.train(
            input_data_paths,
            ivfpq_dir_path,
            timer,
        )
        timer.pop()

        # pax({
        #     "input_data_paths" : input_data_paths,
        #     "ivf_output_data_paths" : ivf_output_data_paths,
        # })
        # print_seq(ivfpq_output_data_paths)

        timer.push("hnsw")
        hnsw_output_data_paths = self.hnsw.train(
            input_data_paths,
            ivfpq_output_data_paths,
            hnsw_dir_path,
            timer,
        )
        timer.pop()

        # pax(0, {
        #     "hnsw_output_data_paths" : hnsw_output_data_paths,
        #     "hnsw_output_data_paths / 0" : hnsw_output_data_paths[0],
        # })
        # print_seq(hnsw_output_data_paths)

        # timer.push("residual")
        # residual_output_data_paths = self.ivfpq.compute_residuals(
        #     # input_data_paths, # 'hnsw_output_data_paths' now has dicts
        #     hnsw_output_data_paths,
        #     ivfpq_dir_path,
        #     timer,
        #     "train",
        # )
        # timer.pop()

        # pax({
        #     "input_data_paths" : input_data_paths,
        #     "ivf_output_data_paths" : ivf_output_data_paths,
        #     "hnsw_output_data_paths" : hnsw_output_data_paths,
        #     "residual_output_data_paths" : residual_output_data_paths,
        #     "residual_output_data_paths / 0" : residual_output_data_paths[0],
        # })

        # return ivf_output_data_paths
        return hnsw_output_data_paths
        # return residual_output_data_paths

    def add(self, input_data_paths, dir_path, timer):

        ivfpq_dir_path = utils.make_sub_dir(dir_path, "ivfpq")
        hnsw_dir_path = utils.make_sub_dir(dir_path, "hnsw")

        timer.push("hnsw")
        hnsw_output_data_paths = self.hnsw.add(
            input_data_paths,
            hnsw_dir_path,
            timer,
        )
        timer.pop()

        # pax(0, {
        #     "hnsw_output_data_paths" : hnsw_output_data_paths,
        #     "hnsw_output_data_paths / 0" : hnsw_output_data_paths[0],
        # })

        # timer.push("residual")
        # residual_output_data_paths = self.ivf.compute_residuals(
        #     # input_data_paths,
        #     hnsw_output_data_paths,
        #     ivf_dir_path,
        #     timer,
        #     "add",
        # )
        # timer.pop()

        timer.push("ivfpq")
        ivfpq_output_data_paths = self.ivfpq.add(
            hnsw_output_data_paths,
            ivfpq_dir_path,
            timer,
            # "add",
        )
        timer.pop()

        # pax({
        #     "input_data_paths" : input_data_paths,
        #     "hnsw_output_data_paths" : hnsw_output_data_paths,
        #     "residual_output_data_paths" : residual_output_data_paths,
        # })
        # print_seq(residual_output_data_paths)
        # print_seq(ivfpq_output_data_paths)

        # return hnsw_output_data_paths
        # return residual_output_data_paths
        return ivfpq_output_data_paths

# eof
