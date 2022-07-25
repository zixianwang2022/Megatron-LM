# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
# from collections import defaultdict
# import faiss
import h5py
# import json
# import numpy as np
import os
# import re
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
        self.nbits = 8

    def dout(self):
        return self.m

    def _train_rank_0(self, input_data_paths, dir_path, timer):

        empty_index_path = self.get_empty_index_path(dir_path)

        if os.path.isfile(empty_index_path):
            return None

        residual_data_paths = [ p["residuals"] for p in input_data_paths ]

        timer.push("load-data")
        input_data = utils.load_data(residual_data_paths, timer)["residuals"]
        timer.pop()

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

    def train(self, input_data_paths, dir_path, timer):

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            timer.push("train")
            self._train_rank_0(input_data_paths, dir_path, timer)
            timer.pop()

        torch.distributed.barrier()

    # def _get_meta_index_paths(self, dir_path, prefix):
    #     meta_path = os.path.join(dir_path, "%s_metas.json" % prefix)
    #     index_path = os.path.join(dir_path, "%s.faissindex" % prefix)
    #     meta_exists = os.path.isfile(meta_path)
    #     index_exists = os.path.isfile(index_path)
    #     assert meta_exists == index_exists
    #     return meta_path, index_path, meta_exists
    # def get_full_paths(self, dir_path):
    #     return self._get_meta_index_paths(dir_path, "full")
    # def get_partial_paths(self, dir_path, rank = None):
    #     rank = torch.distributed.get_rank() if rank is None else rank
    #     return self._get_meta_index_paths(dir_path, "partial_%d" % rank)

    # def load_metas(self, path):
    #     if not os.path.isfile(path):
    #         return set()
    #     else:
    #         with open(path, "r") as f:
    #             return json.load(f)
    # def load_index(self, path, empty_path):
    #     if not path or not os.path.isfile(path):
    #         return faiss.IndexIDMap(faiss.read_index(empty_path))
    #     else:
    #         return faiss.read_index(path)

    def write_partial_index(self, dir_path, pq, new_metas, timer):

        # print_seq(new_metas)

        # Batch, index paths.
        meta_path, index_path, partial_exists = self.get_partial_paths(dir_path)

        # Existing, new batch ids.
        if partial_exists:
            raise Exception("existing batch ids.")
            with open(meta_path, "r") as f:
                existing_metas = json.load(f)
            pax({"existing_metas": existing_metas})
            os.remove(meta_path)
            os.remove(index_path)
        else:
            existing_metas = []

        # batch_ids = existing_batch_ids | new_batch_ids
        # new_metas = [ {batch_id, vec_range} for m in new_metas ] # copy subset
        existing_batch_ids = set(m["batch_id"] for m in existing_metas)
        new_batch_ids = set(m["batch_id"] for m in new_metas)
        assert not (existing_batch_ids & new_batch_ids), "no batch overlap."

        # print_seq(list(new_batch_ids))

        # Write index, batch ids.
        faiss.write_index(pq, index_path)
        with open(meta_path, "w") as f:
            json.dump(list(new_metas), f, indent = 4)

        # Debug.
        # print_seq(added_items)
        # print_seq([ batch_id_path, index_path ])
        # print_seq([ existing_batch_ids, new_batch_ids ])
        print_seq([ existing_metas, new_metas ])

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

        # existing_batch_ids = self.get_existing_batch_ids(dir_path)
        full_meta_path, _, _ = self.get_full_paths(dir_path)
        existing_metas = self.load_metas(full_meta_path)
        existing_batch_ids = set([ m["batch_id"] for m in existing_metas ])

        # pax(0, {
        #     "vec_ranges" : vec_ranges,
        #     "full_batch_id_path" : full_batch_id_path,
        #     "existing_batch_ids" : existing_batch_ids,
        # })
        if existing_metas:
            print_seq(existing_metas)

        missing_batch_ids = []
        missing_count = 0
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        for batch_id in range(len(input_data_paths)):

            if batch_id in existing_batch_ids:
                raise Exception("existing batch id.")
                continue

            if missing_count % world_size == rank:
                missing_batch_ids.append(batch_id)
            missing_count += 1

        # missing_items = [
        #     (bi, vec_id_starts[bi], input_data_paths[bi])
        #     for bi in missing_batch_ids
        # ]
        missing_metas = [ {
            "batch_id" : bi,
            "vec_range" : vec_ranges[bi],
            "input_path" : input_data_paths[bi],
        } for bi in missing_batch_ids ]

        # print_seq(str(vec_id_starts))
        # print_seq("existing_batch_ids = %s." % existing_batch_ids)
        # print_seq("missing_batch_ids = %s." % missing_batch_ids)
        # print_seq(missing_metas)

        return missing_metas

    def create_partial_indexes(self, input_data_paths, dir_path, timer):

        missing_input_data_metas = self.get_missing_input_data_metas(
            input_data_paths,
            dir_path,
            timer,
        )

        # print_seq(missing_input_data_metas)

        # timer.push("add")

        for meta_index, meta in enumerate(missing_input_data_metas):

            timer.push("load-data")
            input_data_path = meta["input_path"]["residuals"]
            input_data = utils.load_data([input_data_path], timer)["residuals"]
            timer.pop()

            print_rank("pqs / add,  batch %d / %d. [ %d vecs ]" % (
                meta_index,
                len(missing_input_data_metas),
                len(input_data),
            ))

            timer.push("init")
            pq = self.load_index(None, empty_index_path)
            self.c_verbose(pq, True)
            timer.pop()

            # print_seq(str(input_data.shape))

            timer.push("add")
            pq.add_with_ids(input_data, np.arange(*meta["vec_range"]))
            timer.pop()

            timer.push("write-partial")
            # faiss.write_index(pq, full_index_path)
            pq = self.write_partial_index(
                dir_path,
                pq,
                [ meta ],
                timer,
            )
            timer.pop()

        # timer.pop()

    def merge_partial_indexes(self, dir_path, timer):

        raise Exception("ready to merge?")

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:

            # Partial meta, index pairs.
            partial_path_pairs = self.get_partial_path_pairs(dir_path)

            # pax({"partial_path_pairs": partial_path_pairs})

            # Load full [ but only if necessary ].
            if partial_path_pairs:

                empty_index_path = self.get_empty_index_path(dir_path)
                full_meta_path, full_index_path, full_exists = \
                    self.get_full_paths(dir_path)

                full_metas = self.load_metas(full_meta_path)
                full_index = self.load_index(full_index_path, empty_index_path)

                # pax({
                #     "empty_index_path" : empty_index_path,
                #     "full_meta_path" : full_meta_path,
                #     "full_index_path" : full_index_path,
                #     "full_exists" : full_exists,
                #     "full_metas" : full_metas,
                #     "full_index" : full_index,
                # })

            for partial_index, partial_path_pair in \
                enumerate(partial_path_pairs.values()):

                partial_meta_path = partial_path_pair["meta"]
                partial_index_path = partial_path_pair["index"]
                partial_metas = self.load_metas(partial_meta_path)
                partial_index = self.load_index(
                    partial_index_path,
                    empty_index_path,
                )

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
                
        torch.distributed.barrier()

    def add(self, input_data_paths, dir_path, timer):

        # empty_index_path = self.get_empty_index_path(dir_path)
        full_index_path = self.get_full_index_path(dir_path)

        if os.path.isfile(full_index_path):
            raise Exception("full index exists.")
            return

        timer.push("create-partials")
        self.create_partial_indexes(input_data_paths, dir_path, timer)
        timer.pop()
        
        raise Exception("ready to merge?")

        timer.push("merge-partials")
        self.merge_partial_indexes(dir_path, timer)
        timer.pop()

# eof
