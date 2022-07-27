# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
# from collections import defaultdict
import faiss
import h5py
# import json
import numpy as np
import os
import re
import torch

from lutil import pax, print_rank, print_seq

from retrieval.index import Index
import retrieval.utils as utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class PQsIndex(Index):

    def __init__(self, args, d, stage_str):
        super().__init__(args, d)
    
        assert stage_str.startswith("PQ")
        self.m = int(stage_str.replace("PQ", ""))
        # self.nbits = 8
        self.nbits = args.pq_nbits

    def dout(self):
        return self.m

    def _train_rank_0(self, input_data_paths, dir_path, timer):

        assert torch.distributed.get_rank() == 0

        empty_index_path = self.get_empty_index_path(dir_path)

        if os.path.isfile(empty_index_path):
            return None

        residual_data_paths = [ p["residuals"] for p in input_data_paths ]

        timer.push("load-data")
        input_data = utils.load_data(residual_data_paths, timer)["residuals"]
        timer.pop()

        timer.push("init")
        if 0:
            pq = faiss.IndexPQ(self.din(), self.m, self.nbits)
            self.c_verbose(pq, True)
        else:
            pq = faiss.IndexIVFPQ( # ... more verbose than PQ
                faiss.IndexFlat(self.args.ivf_dim),
                self.args.ivf_dim,
                1, # self.args.ncluster,
                self.args.pq_dim,
                self.args.pq_nbits,
            )
            # pax({"pq_nbits": self.args.pq_nbits})
            # pax({"pq": pq, "pq / pq": pq.pq})
            self.c_verbose(pq, True)
            self.c_verbose(pq.quantizer, True)
            # self.c_verbose(pq.clustering_index, True) # throws segfault
        timer.pop()

        timer.push("train")
        pq.train(input_data)
        timer.pop()

        timer.push("save")
        faiss.write_index(pq, empty_index_path)
        timer.pop()

    def train(self, input_data_paths, dir_path, timer):

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            timer.push("train")
            self._train_rank_0(input_data_paths, dir_path, timer)
            timer.pop()

        torch.distributed.barrier()

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

    # def write_partial_index(self, dir_path, pq, new_metas, timer):

    #     # print_seq(new_metas)

    #     # Batch, index paths.
    #     meta_path, index_path, partial_exists = self.get_partial_paths(dir_path)

    #     # Existing, new batch ids.
    #     if partial_exists:
    #         raise Exception("existing batch ids.")
    #         with open(meta_path, "r") as f:
    #             existing_metas = json.load(f)
    #         pax({"existing_metas": existing_metas})
    #         os.remove(meta_path)
    #         os.remove(index_path)
    #     else:
    #         existing_metas = []

    #     # batch_ids = existing_batch_ids | new_batch_ids
    #     # new_metas = [ {batch_id, vec_range} for m in new_metas ] # copy subset
    #     existing_batch_ids = set(m["batch_id"] for m in existing_metas)
    #     new_batch_ids = set(m["batch_id"] for m in new_metas)
    #     assert not (existing_batch_ids & new_batch_ids), "no batch overlap."

    #     # print_seq(list(new_batch_ids))

    #     # Write index, batch ids.
    #     faiss.write_index(pq, index_path)
    #     with open(meta_path, "w") as f:
    #         json.dump(list(new_metas), f, indent = 4)

    #     # Debug.
    #     # print_seq(added_items)
    #     # print_seq([ batch_id_path, index_path ])
    #     # print_seq([ existing_batch_ids, new_batch_ids ])
    #     print_seq([ existing_metas, new_metas ])

    # def get_partial_path_pairs(self, dir_path):

    #     paths = os.listdir(dir_path)
    #     paths = [ p for p in paths if p.startswith("partial") ]
    #     assert len(paths) % 2 == 0 # even count

    #     # >>> tmp
    #     if not paths:
    #         return []
    #     # <<<

    #     pair_map = defaultdict(dict)
    #     for path in paths:
    #         tokens = re.split("_|\.", path)
    #         prefix = "_".join(tokens[:2])
    #         path = os.path.join(dir_path, path)
    #         if path.endswith("json"):
    #             pair_map[prefix]["meta"] = path
    #         elif path.endswith("faissindex"):
    #             pair_map[prefix]["index"] = path
    #         # pax({"tokens": tokens, "prefix": prefix})

    #     for pair in pair_map.items():
    #         assert len(pair) == 2

    #     # pax({
    #     #     "paths" : paths,
    #     #     "pair_map" : pair_map,
    #     # })

    #     return pair_map
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
            f = h5py.File(item["residuals"], "r")
            i0 = 0 if not vec_ranges else vec_ranges[-1][1]
            i1 = i0 + len(f["residuals"])
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
            input_data_path = meta["input_path"]["residuals"]
            input_data = utils \
                .load_data([input_data_path], timer)["residuals"] \
                .astype("f4")
            cluster_id_path = meta["input_path"]["centroid_ids"]
            cluster_ids = utils \
                .load_data([cluster_id_path],timer)["centroid_ids"] \
                .astype("i8")
            timer.pop()

            # timer.push("alloc-c++")
            nvecs = len(input_data)
            codes = np.empty((nvecs, self.m), dtype = "uint8")
            # input_data_cpp = faiss.Float32Vector()
            # cluster_ids_cpp = faiss.Int64Vector()
            # codes_cpp = faiss.UInt8Vector()
            # faiss.copy_array_to_vector(input_data.reshape(-1), input_data_cpp)
            # faiss.copy_array_to_vector(cluster_ids.reshape(-1), cluster_ids_cpp)
            # faiss.copy_array_to_vector(codes.reshape(-1), codes_cpp)
            # timer.pop()

            # pax(0, {
            #     "input_data" : input_data,
            #     "cluster_ids" : cluster_ids,
            #     "cluster_ids / ctypes / data" : cluster_ids.ctypes.data,
            #     # "input_data_cpp" : input_data_cpp,
            #     # "cluster_ids_cpp" : cluster_ids_cpp,
            #     # "codes_cpp" : codes_cpp,
            # })

            print_rank("pqs / add,  batch %d / %d. [ %d vecs ]" % (
                meta_index,
                len(missing_input_data_metas),
                nvecs, # len(input_data),
            ))

            timer.push("init")
            _index = faiss.read_index(empty_index_path)
            index = faiss.IndexIVFPQ(
                faiss.IndexFlat(self.args.ivf_dim),
                self.args.ivf_dim,
                self.args.ncluster,
                self.args.pq_dim,
                self.args.pq_nbits,
            )
            index.pq = _index.pq
            index.is_trained = True
            # pax({"index": index})
            self.c_verbose(index, True)
            timer.pop()

            # >>>
            index.add_c(
                n = 10,
                x = faiss.swig_ptr(np.ascontiguousarray(input_data[:10])),
            )
            pax(0, {"index": index})
            print_seq("hi.")
            # <<<

            timer.push("add")
            # pq.add(input_data)
            # pq.add_with_ids(input_data, np.arange(*meta["vec_range"]))
            # index.encode_vectors(input_data, cluster_ids)
            try:
                index.encode_vectors(
                    # nvecs,
                    np.int64(nvecs),
                    # np.array([nvecs]).astype("i8"),
                    input_data, # .ctypes.data,
                    cluster_ids, # .ctypes.data,
                    codes.ctypes, # .data,
                    # False,
                )
                # index.encode_vectors(
                #     nvecs,
                #     input_data_cpp, # .data(),
                #     cluster_ids_cpp, # .data(),
                #     codes_cpp, # .data(),
                # )
            except:
                pax(0, {
                    "nvecs" : nvecs,
                    "input_data" : str(input_data.dtype),
                    "cluster_ids" : str(cluster_ids.dtype),
                    "codes" : str(codes.dtype),
                })
            timer.pop()

            pax(0, {"index": index})

            raise Exception("vectors encoded.")

            timer.push("write-partial")
            # partial_meta_path, partial_index_path = self.get_partial_index_paths(
            #     dir_path,
            #     meta["batch_id"],
            #     len(input_data_paths),
            # )
            partial_index_path = self.get_partial_index_path(dir_path, meta)
            # print_seq(partial_index_path)
            faiss.write_index(pq, partial_index_path)
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

        if torch.distributed.get_rank() == 0:
            timer.push("merge-partials")
            self.merge_partial_indexes(input_data_paths, dir_path, timer)
            timer.pop()

        torch.distributed.barrier() # unnecessary?

        exit(0)

# eof
