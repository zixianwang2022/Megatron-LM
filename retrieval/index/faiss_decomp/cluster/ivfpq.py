# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import h5py
import numpy as np
import os
import re
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

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # def get_partial_index_path(self, dir_path, meta):
    #     sbatch = int(np.ceil(np.log(meta["nbatches"]) / np.log(10)))
    #     svec = int(np.ceil(np.log(meta["nvecs"]) / np.log(10)))
    #     prefix = "partial_%s_%s-%s" % (
    #         str(meta["batch_id"]).zfill(sbatch),
    #         str(meta["vec_range"][0]).zfill(svec),
    #         str(meta["vec_range"][1]).zfill(svec),
    #     )
    #     # print_seq("prefix = %s." % prefix)
    #     index_path = os.path.join(dir_path, "%s.faissindex" % prefix)
    #     # print_seq([ index_path ])
    #     return index_path

    # def get_existing_partial_index_paths(self, dir_path):
    #     paths = os.listdir(dir_path)
    #     paths = [
    #         os.path.join(dir_path, p)
    #         for p in paths
    #         if p.startswith("partial")
    #     ]
    #     for p in paths:
    #         assert p.endswith("faissindex")
    #     paths.sort() # critical. [ sorts by zero-padded batch ids ]
    #     # pax(0, {"paths": paths})
    #     return paths

    # def get_existing_partial_batch_ids(self, dir_path):

    #     # Partial index paths.
    #     paths = self.get_existing_partial_index_paths(dir_path)

    #     # Batch ids.
    #     batch_ids = set()
    #     for path in paths:
    #         tokens = re.split("_|-", path.split("/")[-1])
    #         assert len(tokens) == 4
    #         batch_id = int(tokens[1])
    #         batch_ids.add(batch_id)

    #     # print_seq(str(batch_ids))

    #     return batch_ids

    # def get_missing_input_data_metas(self, input_data_paths, dir_path, timer):

    #     # vec_id_starts = []
    #     # n = 0
    #     # for item in input_data_paths:
    #     #     vec_id_starts.append(n)
    #     #     f = h5py.File(item["residuals"], "r")
    #     #     n += len(f["residuals"])
    #     #     f.close()
    #     vec_ranges = []
    #     for item in input_data_paths:
    #         f = h5py.File(item["data"], "r")
    #         i0 = 0 if not vec_ranges else vec_ranges[-1][1]
    #         i1 = i0 + len(f["data"])
    #         vec_ranges.append((i0, i1))
    #         f.close()

    #     existing_batch_ids = self.get_existing_partial_batch_ids(dir_path)

    #     # if existing_batch_ids:
    #     #     print_seq("existing_batch_ids = %s." % str(existing_batch_ids))

    #     missing_batch_ids = []
    #     missing_count = 0
    #     world_size = torch.distributed.get_world_size()
    #     rank = torch.distributed.get_rank()
    #     for batch_id in range(len(input_data_paths)):

    #         if batch_id in existing_batch_ids:
    #             continue

    #         if missing_count % world_size == rank:
    #             missing_batch_ids.append(batch_id)
    #         missing_count += 1

    #     missing_metas = [ {
    #         "batch_id" : bi,
    #         "nbatches" : len(input_data_paths),
    #         "vec_range" : vec_ranges[bi],
    #         "nvecs" : vec_ranges[-1][1],
    #         "input_path" : input_data_paths[bi],
    #     } for bi in missing_batch_ids ]

    #     # if existing_batch_ids:
    #     #     print_seq([
    #     #         "existing_batch_ids = %s." % str(existing_batch_ids),
    #     #         "missing_batch_ids = %s." % str(missing_batch_ids),
    #     #         *missing_metas,
    #     #     ])
        
    #     return missing_metas

    # def create_partial_indexes(self, input_data_paths, dir_path, timer):

    #     empty_index_path = self.get_empty_index_path(dir_path)

    #     timer.push("missing-metas")
    #     missing_input_data_metas = self.get_missing_input_data_metas(
    #         input_data_paths,
    #         dir_path,
    #         timer,
    #     )
    #     timer.pop()

    #     # Barrier.
    #     # - In case one process writes new partial index before another
    #     #   process calls 'get_missing_input_data_metas'.
    #     torch.distributed.barrier()

    #     # timer.push("add")
    #     for meta_index, meta in enumerate(missing_input_data_metas):

    #         timer.push("load-data")
    #         input_data_path = meta["input_path"]["data"]
    #         input_data = utils \
    #             .load_data([input_data_path], timer)["data"] \
    #             .astype("f4") # f4, float32, float, np.float32
    #         cluster_id_path = meta["input_path"]["centroid_ids"]
    #         cluster_ids = utils \
    #             .load_data([cluster_id_path],timer)["centroid_ids"] \
    #             .astype("i8") # "i8")
    #         timer.pop()

    #         # pax(0, {
    #         #     "input_data" : input_data,
    #         #     "cluster_ids" : cluster_ids,
    #         #     "input_data_path" : input_data_path,
    #         #     "cluster_id_path" : cluster_id_path,
    #         # })
    #         # print_seq("vec_range = %s." % str(meta["vec_range"]))

    #         nvecs = len(input_data)
    #         print_rank("ivfpq / add / partial,  batch %d / %d [%d]. [ %d vecs ]"%(
    #             meta_index,
    #             len(missing_input_data_metas),
    #             meta["batch_id"],
    #             nvecs,
    #         ))

    #         timer.push("init")
    #         index = faiss.read_index(empty_index_path)
    #         # self.c_verbose(index, True)
    #         # self.c_verbose(index.quantizer, True)
    #         timer.pop()

    #         # pax(0, {"index": index})

    #         timer.push("add")
    #         index.add_core(
    #             n = nvecs,
    #             x = my_swig_ptr(input_data),
    #             xids = my_swig_ptr(np.arange(*meta["vec_range"], dtype = "i8")),
    #             precomputed_idx = my_swig_ptr(cluster_ids),
    #         )
    #         timer.pop()

    #         # pax(0, {"index": index})

    #         timer.push("write-partial")
    #         partial_index_path = self.get_partial_index_path(dir_path, meta)
    #         faiss.write_index(index, partial_index_path)
    #         timer.pop()

    #         # print_seq("index written.")

    #     # timer.pop()

    # # def merge_partial_indexes(self, input_data_paths, dir_path, timer):
    # #     '''Merge partial indexes.

    # #     Only run this method on rank 0.
    # #     '''

    # #     assert torch.distributed.get_rank() == 0

    # #     # Index paths.
    # #     empty_index_path = self.get_empty_index_path(dir_path)
    # #     full_index_path = self.get_empty_index_path(dir_path)
    # #     partial_index_paths = self.get_existing_partial_index_paths(dir_path)

    # #     # Full index. (set full's PQ from empty's PQ)
    # #     full_index = faiss.read_index(empty_index_path)
    # #     full_invlists = full_index.invlists
        
    # #     # Add partial indexes.
    # #     for batch_id, partial_index_path in enumerate(partial_index_paths):

    # #         partial_index = faiss.read_index(partial_index_path)
    # #         partial_invlists = partial_index.invlists

    # #         print_rank("ivfpq / add / merge,  batch %d / %d. [ %d vecs ]" % (
    # #             batch_id,
    # #             len(partial_index_paths),
    # #             partial_index.ntotal,
    # #         ))

    # #         for list_id in range(partial_invlists.nlist):
    # #             full_invlists.add_entries(
    # #                 list_id,
    # #                 partial_invlists.list_size(list_id),
    # #                 partial_invlists.get_ids(list_id),
    # #                 partial_invlists.get_codes(list_id),
    # #             )

    # #         full_index.ntotal += partial_index.ntotal

    # #         # pax({
    # #         #     "partial_index" : partial_index,
    # #         #     "partial_invlists" : partial_invlists,
    # #         #     "full_invlists" : full_invlists,
    # #         # })

    # #     pax({"full_index": full_index})
    # def merge_partial_indexes(self, input_data_paths, dir_path, timer):

    #     raise Exception("hi.")
                
    # def add(self, input_data_paths, dir_path, timer):

    #     # empty_index_path = self.get_empty_index_path(dir_path)
    #     full_index_path = self.get_full_index_path(dir_path)

    #     if os.path.isfile(full_index_path):
    #         raise Exception("full index exists.")
    #         return

    #     torch.distributed.barrier() # unnecessary?

    #     timer.push("create-partials")
    #     self.create_partial_indexes(input_data_paths, dir_path, timer)
    #     timer.pop()
        
    #     torch.distributed.barrier() # unnecessary?

    #     # print_seq("created partial indexes.")

    #     # if torch.distributed.get_rank() == 0:
    #     timer.push("merge-partials")
    #     self.merge_partial_indexes(input_data_paths, dir_path, timer)
    #     timer.pop()

    #     torch.distributed.barrier() # unnecessary?

    #     exit(0)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_partial_index_path_map(self, input_data_paths, dir_path, row, col):

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        num_batches = len(input_data_paths)

        batch_str_len = int(np.ceil(np.log(num_batches) / np.log(10))) + 1
        zf = lambda b : str(b).zfill(batch_str_len)

        # First batch id.
        batch_id_0 = 2**row * (rank + col * world_size)

        if batch_id_0 >= num_batches:
            return None

        # Other batch ids.
        if row == 0:
            batch_id_1 = batch_id_2 = batch_id_3 = batch_id_0
        else:
            batch_full_range = 2**row
            batch_half_range = int(batch_full_range / 2)
            batch_id_1 = batch_id_0 + batch_half_range - 1
            batch_id_2 = batch_id_1 + 1
            batch_id_3 = batch_id_2 + batch_half_range - 1

        # Batch ranges.
        def get_batch_range(b0, b1):
            if b1 >= num_batches:
                b1 = num_batches - 1
            if b0 >= num_batches:
                return None
            return b0, b1

        def batch_range_to_index_path(_row, _range):
            if _range is None:
                return None
            else:
                return os.path.join(
                    dir_path,
                    # "partial_r%d_%s-%s.faissindex" % (
                    #     _row,
                    #     *[zf(b) for b in _range],
                    # ),
                    "partial_%s_%s-%s.faissindex" % (
                        # _row, # ... using row id disallows cross-row sharing
                        zf(_range[-1] - _range[0] + 1),
                        *[zf(b) for b in _range],
                    ),
                )

        input_batch_ranges = [
            get_batch_range(batch_id_0, batch_id_1),
            get_batch_range(batch_id_2, batch_id_3),
        ]
        output_batch_range = get_batch_range(batch_id_0, batch_id_3)

        # Path map.
        path_map = {
            "batch_id" : batch_id_0,
            "num_batches" : num_batches,
            "output_index_path" :
            batch_range_to_index_path(row, output_batch_range),
        }
        if row == 0:
            path_map["input_data_path"] = input_data_paths[batch_id_0]
        else:
            input_index_paths = \
                [batch_range_to_index_path(row-1, r) for r in input_batch_ranges]
            input_index_paths = [ p for p in input_index_paths if p is not None ]

            # print_seq(input_index_paths)
            # if len(input_index_paths) < 2:
            #     pax({"input_index_paths": input_index_paths})

            if not input_index_paths:
                return None

            path_map["input_index_paths"] = input_index_paths

        # Return.
        return path_map

    # def add_partial(self, input_data_paths, dir_path, timer, row, col):
    # def add_partial(self, partial_index_path_map, timer):
    # def init_partial(self, partial_index_path_map, dir_path, timer):
    # def init_partial_index(self, partial_index_path_map, dir_path, timer):
    def init_partial(self, partial_index_path_map, dir_path, timer):

        # >>>
        # row = 2; col = 2
        # row = 3; col = 1
        # <<<

        empty_index_path = self.get_empty_index_path(dir_path)
        partial_index_path = partial_index_path_map["output_index_path"]
        input_data_path_item = partial_index_path_map["input_data_path"]

        # print_seq(list(partial_index_path_map.items()))

        if os.path.isfile(partial_index_path):
            return

        timer.push("load-data")
        input_data_path = input_data_path_item["data"]
        input_data = utils \
            .load_data([input_data_path], timer)["data"] \
            .astype("f4") # f4, float32, float, np.float32
        cluster_id_path = input_data_path_item["centroid_ids"]
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

        nvecs = len(input_data)
        print_rank("ivfpq / add / partial,  batch %d / %d. [ %d vecs ]" % (
            partial_index_path_map["batch_id"],
            partial_index_path_map["num_batches"],
            nvecs,
        ))

        timer.push("init")
        index = faiss.read_index(empty_index_path)
        # self.c_verbose(index, True) # with batch_size 1M ... too fast/verbose
        # self.c_verbose(index.quantizer, True)
        timer.pop()

        # pax(0, {"index": index})

        timer.push("add")
        index.add_core(
            n = nvecs,
            x = my_swig_ptr(input_data),
            # xids = my_swig_ptr(np.arange(*meta["vec_range"], dtype = "i8")),
            xids = my_swig_ptr(np.arange(nvecs, dtype = "i8")),
            precomputed_idx = my_swig_ptr(cluster_ids),
        )
        timer.pop()

        # pax(0, {"index": index})

        timer.push("write-partial")
        faiss.write_index(index, partial_index_path)
        timer.pop()

        # print_seq("index written.")

    # def merge_partial_indexes(self, partial_index_path_map, dir_path, timer):
    def merge_partial(self, partial_index_path_map, dir_path, timer):
        '''Merge partial indexes.'''

        # Index paths.
        empty_index_path = self.get_empty_index_path(dir_path)
        output_index_path = partial_index_path_map["output_index_path"]
        input_index_paths = partial_index_path_map["input_index_paths"]

        # print_seq([
        #     empty_index_path,
        #     *input_index_paths,
        #     output_index_path,
        # ])

        if os.path.isfile(output_index_path):
            # raise Exception("merged index exists.")
            return

        assert len(input_index_paths) >= 2, \
            "if singular input index, path should already exist."

        # Output index.
        output_index = faiss.read_index(empty_index_path)
        output_invlists = output_index.invlists
        
        # Merge input indexes.
        for input_iter, input_index_path in enumerate(input_index_paths):

            assert input_index_path is not None, "edge case."
            # if input_index_path is None:
            #     pax({"partial_index_path_map": partial_index_path_map})

            input_index = faiss.read_index(input_index_path)
            input_invlists = input_index.invlists

            print_rank("ivfpq / add / merge, input %d / %d. [ +%d -> %d ]" % (
                input_iter,
                len(input_index_paths),
                input_index.ntotal,
                input_index.ntotal + output_index.ntotal,
            ))

            for list_id in range(input_invlists.nlist):
                output_invlists.add_entries(
                    list_id,
                    input_invlists.list_size(list_id),
                    input_invlists.get_ids(list_id),
                    input_invlists.get_codes(list_id),
                )

            output_index.ntotal += input_index.ntotal

            # pax({
            #     "input_index" : input_index,
            #     "output_index" : output_index,
            #     "input_invlists" : input_invlists,
            #     "output_invlists" : output_invlists,
            # })

        # pax({
        #     "input_index" : input_index,
        #     "output_index" : output_index,
        # })

        timer.push("write-output")
        faiss.write_index(output_index, output_index_path)
        timer.pop()

    def add(self, input_data_paths, dir_path, timer):

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        num_batches = len(input_data_paths)
        # num_batches = 47000 # ... ~15.52 rows
        num_rows = int(np.ceil(np.log(num_batches) / np.log(2))) + 1
        
        for row in range(num_rows):

            timer.push("row %d of %d" % (row, num_rows))

            num_cols = int(np.ceil(num_batches / world_size / 2**row))
            # for col in range(rank, num_batches, world_size * int(2**row)):
            for col in range(num_cols):

                timer.push("col")

                print_rank(0, "r %d / %d, c %d / %d." % (
                    row,
                    num_rows,
                    col,
                    num_cols,
                ))

                # Input/output index paths.
                partial_index_path_map = self.get_partial_index_path_map(
                    input_data_paths,
                    dir_path,
                    row,
                    col,
                )

                # Handle edge cases.
                if partial_index_path_map is None:
                    continue

                # Initialize/merge partial indexes.
                if row == 0:
                    timer.push("init-partial")
                    self.init_partial(partial_index_path_map, dir_path, timer)
                    timer.pop()
                else:
                    timer.push("merge-partial")
                    self.merge_partial(partial_index_path_map, dir_path, timer)
                    timer.pop()

                timer.pop()

            torch.distributed.barrier() # prevent inter-row race condition.

            timer.pop()

            # >>>
            if row == 7:
                print_seq("finished row %d." % row)
            # <<<

        pax(0, {
            "num_batches" : num_batches,
            "num_rows" : num_rows,
        })

        torch.distributed.barrier() # unnecessary?

        exit(0)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# eof
