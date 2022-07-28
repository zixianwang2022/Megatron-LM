# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import h5py
import numpy as np
import os
import torch

from lutil import pax, print_rank, print_seq

import retrieval.utils as utils

from retrieval.index import Index

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def my_swig_ptr(x):
    return faiss.swig_ptr(np.ascontiguousarray(x))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class IVFIndex(Index):
class IVFPQIndex(Index):

    # def __init__(self, args, d, nlist):
    #     super().__init__(args, d)
    #     self.nlist = nlist
    # def __init__(self, args):
    #     super().__init__(args, args.ivf_dim, None)

    # def dout(self):
    #     return self.din()

    # def verbose(self, v):
    #     self.c_verbose(self.ivf, v)
    #     # self.c_verbose(self.quantizer, v)

    @classmethod
    def c_cpu_to_gpu(cls, index):
        # raise Exception("use 'current_device' only.")
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index.d))
        index.clustering_index = clustering_index

    def _train(
            self,
            input_data_paths,
            dir_path,
            timer,
    ):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)

        if os.path.isfile(empty_index_path):
            return

        timer.push("load-data")
        inp = utils.load_data(input_data_paths, timer)["data"]
        timer.pop()

        # pax({"inp": str(inp.shape)})

        timer.push("init")
        # ivf = faiss.IndexIVFFlat(
        #     faiss.IndexFlatL2(self.din()),
        #     self.din(),
        #     self.nlist,
        # )
        index = faiss.IndexIVFPQ(
            faiss.IndexFlat(self.args.ivf_dim),
            self.args.ivf_dim,
            self.args.ncluster,
            self.args.pq_m,
            self.args.pq_nbits,
        )
        self.c_verbose(index, True)
        self.c_verbose(index.quantizer, True)
        self.c_cpu_to_gpu(index)
        self.c_verbose(index.clustering_index, True)
        timer.pop()

        # pax({"index": index})

        timer.push("train")
        index.train(inp)
        timer.pop()

        timer.push("save")
        faiss.write_index(index, empty_index_path)
        timer.pop()

    def get_centroid_data_path(self, dir_path):
        return self.get_output_data_path(dir_path, "train", "centroids")

    def _forward_centroids(
            self,
            input_data_paths,
            dir_path,
            timer,
            task,
    ):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)
        output_data_path = self.get_centroid_data_path(dir_path)

        if not os.path.isfile(output_data_path):

            timer.push("init")
            index = faiss.read_index(empty_index_path)
            self.c_verbose(index, True)
            self.c_verbose(index.quantizer, True)
            # self.c_cpu_to_gpu(index) # ... unnecessary for centroid reconstruct
            # self.c_verbose(index.clustering_index, True) # ... only after gpu
            timer.pop()

            timer.push("save-data")
            centroids = index.quantizer.reconstruct_n(0, self.args.ncluster)
            utils.save_data({"centroids": centroids}, output_data_path)
            timer.pop()

        # pax({ ... })

        return [ output_data_path ]

    def train(self, input_data_paths, dir_path, timer):

        # timer = args[-1]

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:

            timer.push("train")
            self._train(input_data_paths, dir_path, timer)
            timer.pop()

            timer.push("forward")
            output_data_paths = self._forward_centroids(
                input_data_paths,
                dir_path,
                timer,
                "train",
            )
            timer.pop()

        torch.distributed.barrier()

        # pax({"output_data_paths": output_data_paths})

        # return output_data_paths
        return [ self.get_centroid_data_path(dir_path) ]

    # def compute_residuals(
    #         self,
    #         input_data_paths,
    #         # centroid_id_data_paths,
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

    #     all_output_data_paths = [ {
    #         "residuals" : o,
    #         "centroid_ids" : i["centroid_ids"],
    #     } for i, o in zip(input_data_paths, all_output_data_paths) ]

    #     # pax({
    #     #     "all_output_data_paths" : all_output_data_paths,
    #     #     "all_output_data_paths / 0" : all_output_data_paths[0],
    #     #     "missing_output_data_path_map" : missing_output_data_path_map,
    #     # })

    #     if not missing_output_data_path_map:
    #         return all_output_data_paths

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
    #     # for (i0, i1), output_data_path in missing_output_data_path_map.items():
    #     for output_index, (input_index, output_data_path) in \
    #         enumerate(missing_output_data_path_map.items()):

    #         timer.push("load-data")
    #         input_data_path_item = input_data_paths[input_index]
    #         # pax({"input_data_path_item": input_data_path_item})
    #         inp = utils.load_data([ input_data_path_item["data"] ], timer)["data"]
    #         centroid_ids = utils.load_data(
    #             [ input_data_path_item["centroid_ids"] ],
    #             timer,
    #         )["centroid_ids"].astype("i8")
    #         timer.pop()

    #         assert len(inp) == len(centroid_ids)

    #         # pax({
    #         #     "inp" : str(inp.shape),
    #         #     "centroid_ids" : str(centroid_ids.shape),
    #         # })

    #         timer.push("forward-batch")
    #         print_rank("foward batch %d / %d. [ %d vecs ]" % (
    #             output_index,
    #             len(missing_output_data_path_map),
    #             len(inp),
    #         )) # , flush = True)

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
            
    #         expanded_centroids = centroids[np.squeeze(centroid_ids)]
    #         residuals = inp - expanded_centroids
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
    #             "residuals" : residuals,
    #             # "centroid_ids" : centroid_ids, # separate file
    #         }, output_data_path)
    #         timer.pop()

    #         timer.pop()

    #     timer.pop()
        
    #     return all_output_data_paths

    def get_partial_index_path(self, dir_path, meta):
        sbatch = int(np.ceil(np.log(meta["nbatches"]) / np.log(10)))
        svec = int(np.ceil(np.log(meta["nvecs"]) / np.log(10)))
        prefix = "partial_%s_%s-%s" % (
            str(meta["batch_id"]).zfill(sbatch),
            str(meta["vec_range"][0]).zfill(svec),
            str(meta["vec_range"][1]).zfill(svec),
        )
        # print_seq("prefix = %s." % prefix)
        index_path = os.path.join(dir_path, "%s.faissindex" % prefix)
        # print_seq([ index_path ])
        return index_path

    def get_existing_partial_index_paths(self, dir_path):
        paths = os.listdir(dir_path)
        paths = [
            os.path.join(dir_path, p)
            for p in paths
            if p.startswith("partial")
        ]
        for p in paths:
            assert p.endswith("faissindex")
        paths.sort() # critical. [ sorts by zero-padded batch ids ]
        # pax(0, {"paths": paths})
        return paths

    def get_existing_partial_batch_ids(self, dir_path):

        # Partial index paths.
        paths = self.get_existing_partial_index_paths(dir_path)

        # Batch ids.
        batch_ids = set()
        for path in paths:
            tokens = re.split("_|-", path.split("/")[-1])
            assert len(tokens) == 4
            batch_id = int(tokens[1])
            batch_ids.add(batch_id)

        # print_seq(str(batch_ids))

        return batch_ids

    def get_missing_input_data_metas(self, input_data_paths, dir_path, timer):

        # vec_id_starts = []
        # n = 0
        # for item in input_data_paths:
        #     vec_id_starts.append(n)
        #     f = h5py.File(item["residuals"], "r")
        #     n += len(f["residuals"])
        #     f.close()
        vec_ranges = []
        for item in input_data_paths:
            f = h5py.File(item["data"], "r")
            i0 = 0 if not vec_ranges else vec_ranges[-1][1]
            i1 = i0 + len(f["data"])
            vec_ranges.append((i0, i1))
            f.close()

        existing_batch_ids = self.get_existing_partial_batch_ids(dir_path)

        # if existing_batch_ids:
        #     print_seq("existing_batch_ids = %s." % str(existing_batch_ids))

        missing_batch_ids = []
        missing_count = 0
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        for batch_id in range(len(input_data_paths)):

            if batch_id in existing_batch_ids:
                continue

            if missing_count % world_size == rank:
                missing_batch_ids.append(batch_id)
            missing_count += 1

        missing_metas = [ {
            "batch_id" : bi,
            "nbatches" : len(input_data_paths),
            "vec_range" : vec_ranges[bi],
            "nvecs" : vec_ranges[-1][1],
            "input_path" : input_data_paths[bi],
        } for bi in missing_batch_ids ]

        # if existing_batch_ids:
        #     print_seq([
        #         "existing_batch_ids = %s." % str(existing_batch_ids),
        #         "missing_batch_ids = %s." % str(missing_batch_ids),
        #         *missing_metas,
        #     ])
        
        return missing_metas

    def create_partial_indexes(self, input_data_paths, dir_path, timer):

        empty_index_path = self.get_empty_index_path(dir_path)

        timer.push("missing-metas")
        missing_input_data_metas = self.get_missing_input_data_metas(
            input_data_paths,
            dir_path,
            timer,
        )
        timer.pop()

        # Barrier.
        # - In case one process writes new partial index before another
        #   process calls 'get_missing_input_data_metas'.
        torch.distributed.barrier()

        # timer.push("add")
        for meta_index, meta in enumerate(missing_input_data_metas):

            timer.push("load-data")
            input_data_path = meta["input_path"]["data"]
            input_data = utils \
                .load_data([input_data_path], timer)["data"] \
                .astype("f4") # f4, float32, float, np.float32
            cluster_id_path = meta["input_path"]["centroid_ids"]
            cluster_ids = utils \
                .load_data([cluster_id_path],timer)["centroid_ids"] \
                .astype("i8") # "i8")
            timer.pop()

            # pax(0, {
            #     "input_data" : input_data,
            #     "cluster_ids" : cluster_ids,
            #     "input_data_path" : input_data_path,
            #     "cluster_id_path" : cluster_id_path,
            # })
            # print_seq("vec_range = %s." % str(meta["vec_range"]))

            nvecs = len(input_data)
            print_rank("pqs / add,  batch %d / %d [%d]. [ %d vecs ]" % (
                meta_index,
                len(missing_input_data_metas),
                meta["batch_id"],
                nvecs,
            ))

            timer.push("init")
            index = faiss.read_index(empty_index_path)
            # self.c_verbose(index, True)
            # self.c_verbose(index.quantizer, True)
            timer.pop()

            # pax(0, {"index": index})

            timer.push("add")
            index.add_core(
                n = nvecs,
                x = my_swig_ptr(input_data),
                xids = my_swig_ptr(np.arange(*meta["vec_range"], dtype = "i8")),
                precomputed_idx = my_swig_ptr(cluster_ids),
            )
            timer.pop()

            # pax(0, {"index": index})

            timer.push("write-partial")
            partial_index_path = self.get_partial_index_path(dir_path, meta)
            faiss.write_index(index, partial_index_path)
            timer.pop()

            # print_seq("index written.")

        # timer.pop()

    def merge_partial_indexes(self, input_data_paths, dir_path, timer):
        '''Merge partial indexes.

        Only run this method on rank 0.
        '''

        assert torch.distributed.get_rank() == 0

        # Index paths.
        empty_index_path = self.get_empty_index_path(dir_path)
        full_index_path = self.get_empty_index_path(dir_path)
        partial_index_paths = self.get_existing_partial_index_paths(dir_path)

        # Full index. (set full's PQ from empty's PQ)
        empty_index = faiss.read_index(empty_index_path)
        full_index = faiss.IndexIVFPQ(
            faiss.IndexFlat(self.args.ivf_dim),
            self.args.ivf_dim,
            self.args.ncluster,
            self.args.pq_dim,
            self.args.pq_nbits,
        )
        full_index.pq = empty_index.pq
        
        # Add partial indexes.
        for partial_index, partial_index_path in enumerate(partial_index_paths):

            # Verify
            partial_batch_id =int(partial_index_path.split("/")[-1].split("_")[1])
            assert partial_index == partial_batch_id, "sanity check."

            partial_cluster_id_data_path = \
                input_data_paths[partial_index]["centroid_ids"]
            f = h5py.File(partial_cluster_id_data_path, "r")
            partial_cluster_ids = np.copy(f["centroid_ids"])
            f.close()

            partial_index = faiss.read_index(partial_index_path)
            partial_list_size = len(partial_cluster_ids)
            partial_ids = partial_index.invlists.get_ids(0)
            partial_codes = partial_index.invlists.get_codes(0)

            full_index.add_core()

            pax({
                "partial_index_path" : partial_index_path,
                "partial_batch_id" : partial_batch_id,
                "partial_cluster_id_data_path" : partial_cluster_id_data_path,
                "partial_cluster_ids" : partial_cluster_ids,
                "partial_index" : partial_index,
                "partial_list_size" : partial_list_size,
                "partial_ids" : partial_ids,
                "partial_codes" : partial_codes,
            })

            # partial_codes = np.reshape(
            #     faiss.vector_to_array(partial_index.codes),
            #     (partial_index.ntotal, -1),
            # )
            # partial_index.copy_subset_to()
            # full_index.add_entries()
            # full_index.add_codes()
            # full_index.add_core()
            full_index.invlists.add_entries()
            # for i in range(32):
            #     full_index.codes.push_back(partial_index.codes.at(i))

            pax({
                # "partial_index / codes" : partial_codes,
                "full_index" : full_index,
                "partial_index" : partial_index,
                # "full_index / codes / size" : full_index.codes.size(),
                # "partial_index / codes / size" : partial_index.codes.size(),
            })

            id_map = partial_index.id_map
            # id_map_ptr = &id_map[0]
            id_map_py = faiss.vector_to_array(partial_index.id_map)

            pax({
                "partial_path_pair" : partial_path_pair,
                "partial_meta_path" : partial_meta_path,
                "partial_index_path" : partial_index_path,
                "partial_metas" : partial_metas,
                "partial_index" : partial_index,
                "partial_index / index" : partial_index.index,
                "id_map" : id_map,
                "id_map / size" : id_map.size(),
                "id_map_py" : id_map_py,
            })
                
    def add(self, input_data_paths, dir_path, timer):

        # empty_index_path = self.get_empty_index_path(dir_path)
        full_index_path = self.get_full_index_path(dir_path)

        if os.path.isfile(full_index_path):
            raise Exception("full index exists.")
            return

        torch.distributed.barrier() # unnecessary?

        timer.push("create-partials")
        self.create_partial_indexes(input_data_paths, dir_path, timer)
        timer.pop()
        
        torch.distributed.barrier() # unnecessary?

        print_seq("created partial indexes.")

        if torch.distributed.get_rank() == 0:
            timer.push("merge-partials")
            self.merge_partial_indexes(input_data_paths, dir_path, timer)
            timer.pop()

        torch.distributed.barrier() # unnecessary?

        exit(0)

# eof
