# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import numpy as np
import os
import torch

from lutil import pax, print_rank, print_seq

import retrieval.utils as utils

from retrieval.index import Index

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class IVFIndex(Index):

    def __init__(self, args, d, nlist):
        super().__init__(args, d)
        self.nlist = nlist

    def dout(self):
        return self.din()

    # def verbose(self, v):
    #     self.c_verbose(self.ivf, v)
    #     # self.c_verbose(self.quantizer, v)

    @classmethod
    def c_cpu_to_gpu(cls, ivf):
        raise Exception("use 'current_device' only.")
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(ivf.d))
        ivf.clustering_index = clustering_index

    def _train(
            self,
            input_data_paths,
            dir_path,
            timer,
    ):

        empty_index_path = self.get_empty_index_path(dir_path)

        if os.path.isfile(empty_index_path):
            return

        timer.push("load-data")
        inp = utils.load_data(input_data_paths, timer)["data"]
        timer.pop()

        # pax({"inp": str(inp.shape)})

        timer.push("init")
        ivf = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(self.din()),
            self.din(),
            self.nlist,
        )
        self.c_verbose(ivf, True)
        self.c_verbose(ivf.quantizer, True)
        self.c_cpu_to_gpu(ivf)
        self.c_verbose(ivf.clustering_index, True)
        timer.pop()

        timer.push("train")
        ivf.train(inp)
        timer.pop()

        timer.push("save")
        faiss.write_index(ivf, empty_index_path)
        timer.pop()

    def _forward_centroids(
            self,
            input_data_paths,
            dir_path,
            timer,
            task,
    ):

        empty_index_path = self.get_empty_index_path(dir_path)
        output_data_path = self.get_output_data_path(dir_path,"train","centroids")

        if not os.path.isfile(output_data_path):

            timer.push("init")
            ivf = faiss.read_index(empty_index_path)
            self.c_verbose(ivf, True)
            self.c_verbose(ivf.quantizer, True)
            # self.c_cpu_to_gpu(ivf) # ... unnecessary for centroid reconstruct
            # self.c_verbose(ivf.clustering_index, True) # ... only after gpu
            timer.pop()

            timer.push("save-data")
            centroids = ivf.quantizer.reconstruct_n(0, self.nlist)
            utils.save_data({"centroids": centroids}, output_data_path)
            timer.pop()

        # pax({ ... })

        return [ output_data_path ]

    def train(self, *args):

        timer = args[-1]

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:

            timer.push("train")
            self._train(*args)
            timer.pop()

            timer.push("forward")
            output_data_paths = self._forward_centroids(*args, "train")
            timer.pop()

        torch.distributed.barrier()

        # pax({"output_data_paths": output_data_paths})

        return output_data_paths

    # def compute_residuals(
    #         self,
    #         input_data_paths,
    #         centroid_id_data_paths,
    #         dir_path,
    #         timer,
    #         task,
    # ):

    #     empty_index_path = self.get_empty_index_path(dir_path)
    #     # output_data_path = self.get_output_data_path(dir_path, task, "res");

    #     all_output_data_paths, missing_output_data_path_map = \
    #         self.get_missing_output_data_path_map(
    #             input_data_paths,
    #             dir_path,
    #             task + "-res",
    #         )

    #     # pax({
    #     #     "all_output_data_paths" : all_output_data_paths,
    #     #     "missing_output_data_path_map" : missing_output_data_path_map,
    #     # })

    #     if not missing_output_data_path_map:
    #         return all_output_data_paths

    #     # >>>
    #     timer.push("load-data")
    #     inp = utils.load_data(input_data_paths, timer)["data"]
    #     centroid_ids = utils.load_data(
    #         centroid_id_data_paths,
    #         timer,
    #     )["centroid_ids"].astype("i8")
    #     timer.pop()

    #     # pax({"inp": inp, "centroid_ids": centroid_ids})

    #     timer.push("init")
    #     ivf = faiss.read_index(empty_index_path)
    #     self.c_verbose(ivf, True)
    #     self.c_verbose(ivf.quantizer, True)
    #     # self.c_cpu_to_gpu(ivf) # ... unnecessary for centroid reconstruct
    #     # self.c_verbose(ivf.clustering_index, True) # ... only after gpu
    #     timer.pop()

    #     centroids = ivf.quantizer.reconstruct_n(0, self.nlist)

    #     # pax({
    #     #     "ivf" : ivf,
    #     #     "centroids" : centroids,
    #     # })

    #     timer.push("forward-batches")
    #     for (i0, i1), output_data_path in missing_output_data_path_map.items():

    #         timer.push("forward-batch")
    #         ntrain = self.args.ntrain
    #         print("foward batch [%d:%d] / %d." % (i0, i1, ntrain), flush = True)

    #         sub_inp = inp[i0:i1]
    #         sub_centroid_ids = centroid_ids[i0:i1]

    #         # pax({
    #         #     "i0" : i0,
    #         #     "i1" : i1,
    #         #     "output_data_path" : output_data_path,
    #         #     "inp" : str(inp.shape),
    #         #     "sub_inp" : str(sub_inp.shape),
    #         # })

    #         timer.push("residual")
    #         # >>>
    #         # res = np.zeros_like(inp)
    #         # # pax({"res": res, "centroid_ids": centroid_ids})
    #         # # pax({"inp": inp})
    #         # # ivf.compute_residual_n()
    #         # try:
    #         #     ivf.compute_residual_n(len(inp), inp, res, centroid_ids)
    #         # except Exception as e:
    #         #     pax({
    #         #         "ivf" : ivf,
    #         #         "len(inp)" : len(inp),
    #         #         "inp" : inp,
    #         #         "res" : res,
    #         #         "centroid_ids" : centroid_ids,
    #         #         "e" : e,
    #         #     })
    #         # # ivf.compute_residual_n(inp, res, centroid_ids)
    #         # pax({"res": res})
    #         # +++
            
    #         sub_expanded_centroids = centroids[np.squeeze(sub_centroid_ids)]
    #         sub_residuals = sub_inp - sub_expanded_centroids
    #         # pax({
    #         #     "centroids" : centroids,
    #         #     "sub_inp" : sub_inp,
    #         #     "sub_centroid_ids" : sub_centroid_ids,
    #         #     "sub_expanded_centroids" : sub_expanded_centroids,
    #         #     "sub_residuals" : sub_residuals,
    #         #     "output_data_path" : output_data_path,
    #         # })
    #         # <<<
    #         timer.pop()

    #         timer.push("save-data")
    #         utils.save_data({
    #             "residuals" : sub_residuals,
    #             "centroid_ids" : sub_centroid_ids,
    #         }, output_data_path)
    #         timer.pop()

    #         timer.pop()

    #     timer.pop()
        
    #     return all_output_data_paths
    def compute_residuals(
            self,
            input_data_paths,
            # centroid_id_data_paths,
            dir_path,
            timer,
            task,
    ):

        empty_index_path = self.get_empty_index_path(dir_path)
        # output_data_path = self.get_output_data_path(dir_path, task, "res");

        all_output_data_paths, missing_output_data_path_map = \
            self.get_missing_output_data_path_map(
                input_data_paths,
                dir_path,
                task + "-res",
            )

        all_output_data_paths = [ {
            "residuals" : o,
            "centroid_ids" : i["centroid_ids"],
        } for i, o in zip(input_data_paths, all_output_data_paths) ]

        # pax({
        #     "all_output_data_paths" : all_output_data_paths,
        #     "all_output_data_paths / 0" : all_output_data_paths[0],
        #     "missing_output_data_path_map" : missing_output_data_path_map,
        # })

        if not missing_output_data_path_map:
            return all_output_data_paths

        timer.push("init")
        ivf = faiss.read_index(empty_index_path)
        self.c_verbose(ivf, True)
        self.c_verbose(ivf.quantizer, True)
        # self.c_cpu_to_gpu(ivf) # ... unnecessary for centroid reconstruct
        # self.c_verbose(ivf.clustering_index, True) # ... only after gpu
        timer.pop()

        centroids = ivf.quantizer.reconstruct_n(0, self.nlist)

        # pax({
        #     "ivf" : ivf,
        #     "centroids" : centroids,
        # })

        timer.push("forward-batches")
        # for (i0, i1), output_data_path in missing_output_data_path_map.items():
        for output_index, (input_index, output_data_path) in \
            enumerate(missing_output_data_path_map.items()):

            timer.push("load-data")
            input_data_path_item = input_data_paths[input_index]
            # pax({"input_data_path_item": input_data_path_item})
            inp = utils.load_data([ input_data_path_item["data"] ], timer)["data"]
            centroid_ids = utils.load_data(
                [ input_data_path_item["centroid_ids"] ],
                timer,
            )["centroid_ids"].astype("i8")
            timer.pop()

            assert len(inp) == len(centroid_ids)

            # pax({
            #     "inp" : str(inp.shape),
            #     "centroid_ids" : str(centroid_ids.shape),
            # })

            timer.push("forward-batch")
            print_rank("foward batch %d / %d. [ %d vecs ]" % (
                output_index,
                len(missing_output_data_path_map),
                len(inp),
            )) # , flush = True)

            timer.push("residual")
            # >>>
            # res = np.zeros_like(inp)
            # # pax({"res": res, "centroid_ids": centroid_ids})
            # # pax({"inp": inp})
            # # ivf.compute_residual_n()
            # try:
            #     ivf.compute_residual_n(len(inp), inp, res, centroid_ids)
            # except Exception as e:
            #     pax({
            #         "ivf" : ivf,
            #         "len(inp)" : len(inp),
            #         "inp" : inp,
            #         "res" : res,
            #         "centroid_ids" : centroid_ids,
            #         "e" : e,
            #     })
            # # ivf.compute_residual_n(inp, res, centroid_ids)
            # pax({"res": res})
            # +++
            
            expanded_centroids = centroids[np.squeeze(centroid_ids)]
            residuals = inp - expanded_centroids
            # pax({
            #     "centroids" : centroids,
            #     "sub_inp" : sub_inp,
            #     "sub_centroid_ids" : sub_centroid_ids,
            #     "sub_expanded_centroids" : sub_expanded_centroids,
            #     "sub_residuals" : sub_residuals,
            #     "output_data_path" : output_data_path,
            # })
            # <<<
            timer.pop()

            timer.push("save-data")
            utils.save_data({
                "residuals" : residuals,
                # "centroid_ids" : centroid_ids, # separate file
            }, output_data_path)
            timer.pop()

            timer.pop()

        timer.pop()
        
        return all_output_data_paths

# eof
