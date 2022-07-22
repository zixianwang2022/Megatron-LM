# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import h5py
import json
import numpy as np
import os
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
        input_data = utils.load_data(residual_data_paths, timer)["residuals"]
        # centroid_ids = utils.load_data(centroid_id_data_paths, timer)["centroid_ids"]
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

    def train(self, input_data_paths, dir_path, timer):

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            timer.push("train")
            self._train_rank_0(input_data_paths, dir_path, timer)
            timer.pop()

        torch.distributed.barrier()

    # def add(self, input_data_paths, dir_path, timer):

    #     empty_index_path = self.get_empty_index_path(dir_path)
    #     full_index_path = self.get_full_index_path(dir_path)

    #     if os.path.isfile(full_index_path):
    #         return None

    #     all_output_data_paths, missing_output_data_path_map = \
    #         self.get_missing_output_data_path_map(
    #             input_data_paths,
    #             dir_path,
    #             "add",
    #         )

    #     print_seq(list(missing_output_data_path_map.values()))

    #     if not missing_output_data_path_map:
    #         raise Exception("merge full index.")

    #     timer.push("init")
    #     pq = faiss.read_index(empty_index_path)
    #     self.c_verbose(pq, True)
    #     timer.pop()

    #     # pax({"pq": pq})

    #     # timer.push("add")

    #     for input_index, input_data_path_item in enumerate(input_data_paths):

    #         timer.push("load-data")
    #         input_data_path = input_data_path_item["residuals"]
    #         # centroid_id_data_path = input_data_path_item["centroid_ids"]
    #         input_data = utils.load_data([ input_data_path ], timer)["residuals"]
    #         # centroid_ids = utils.load_data([ centroid_id_data_path ], timer)["centroid_ids"]
    #         # assert len(input_data) == len(centroid_ids)
    #         timer.pop()

    #         # pax({
    #         #     "input_data_path" : input_data_path,
    #         #     "centroid_id_data_path" : centroid_id_data_path,
    #         #     "input_data" : str(input_data.shape),
    #         #     "centroid_ids" : str(centroid_ids.shape),
    #         # })

    #         print("pqs / add,  batch %d / %d. [ %d vecs ]" % (
    #             input_index,
    #             len(input_data_paths),
    #             len(input_data),
    #         ), flush = True)

    #         timer.push("add")
    #         pq.add(input_data)
    #         timer.pop()

    #     timer.push("save")
    #     faiss.write_index(pq, full_index_path)
    #     timer.pop()

    #     # timer.pop()

    # def get_full_batch_id_path(self, dir_path):
    def get_full_paths(self, dir_path):
        return (
            os.path.join(dir_path, "full_batch_ids.json"),
            self.get_full_index_path(dir_path),
        )
    # def get_partial_batch_id_path(self, dir_path):
    def get_partial_paths(self, dir_path):
        rank = torch.distributed.get_rank()
        return (
            os.path.join(dir_path, "partial_%d_batch_ids.json" % rank),
            os.path.join(dir_path, "partial_%d.faissindex" % rank),
        )

    def write_partial_index(self, dir_path, pq, new_items, timer):

        # Batch, index paths.
        batch_id_path, index_path = self.get_partial_paths(dir_path)

        batch_id_path_exists = os.path.isfile(batch_id_path)
        index_path_exists = os.path.isfile(index_path)
        assert batch_id_path_exists == index_path_exists
        
        # Existing, new batch ids.
        if batch_id_path_exists:
            raise Exception("existing batch ids.")
            existing_batch_ids = json.load(batch_id_path)
            os.remove(batch_id_path)
            os.remove(index_path)
        else:
            existing_batch_ids = set()

        new_batch_ids = set(a[0] for a in new_items)
        batch_ids = existing_batch_ids | new_batch_ids

        # Write index, batch ids.
        faiss.write_index(pq, index_path)
        with open(batch_id_path, "w") as f:
            json.dump(list(batch_ids), f)

        # print_seq(added_items)
        # print_seq([ batch_id_path, index_path ])
        print_seq([ existing_batch_ids, new_batch_ids ])

    def merge_partial_indexes(self, dir_path, timer):

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:

            print_rank("do something here.")

        torch.distributed.barrier()

    def get_existing_batch_ids(self, dir_path):

        batch_id_path, _ = self.get_full_paths(dir_path)
        if not os.path.isfile(batch_id_path):
            return set()

        raise Exception("full batch ids exist.")
        
        print_seq("batch_id_path = '%s'." % batch_id_path)

        return existing_batch_ids

    # def get_missing_input_data_path_map(self, input_data_paths, dir_path, timer):
    def get_missing_input_data_items(self, input_data_paths, dir_path, timer):

        vec_id_starts = []
        n = 0
        for item in input_data_paths:
            vec_id_starts.append(n)
            f = h5py.File(item["residuals"], "r")
            n += len(f["residuals"])
            f.close()

        existing_batch_ids = self.get_existing_batch_ids(dir_path)

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

        missing_items = [
            (bi, vec_id_starts[bi], input_data_paths[bi])
            for bi in missing_batch_ids
        ]

        # print_seq(str(vec_id_starts))
        # print_seq("existing_batch_ids = %s." % existing_batch_ids)
        # print_seq("missing_batch_ids = %s." % missing_batch_ids)
        # print_seq(missing_items)

        return missing_items

    def add(self, input_data_paths, dir_path, timer):

        empty_index_path = self.get_empty_index_path(dir_path)
        # rank_index_path = self.get_rank_index_path(dir_path)
        full_index_path = self.get_full_index_path(dir_path)

        # if os.path.isfile(full_index_path):
        #     ... see missing input paths, below ...
        #     return None

        # >>>
        # all_output_data_paths, missing_output_data_path_map = \
        #     self.get_missing_output_data_path_map(
        #         input_data_paths,
        #         dir_path,
        #         "add",
        #     )

        # print_seq(list(missing_output_data_path_map.values()))

        # if not missing_output_data_path_map:
        #     raise Exception("merge full index.")
        #     return
        # <<<

        timer.push("merge-partials")
        self.merge_partial_indexes(dir_path, timer)
        timer.pop()

        missing_input_data_items = self.get_missing_input_data_items(
            input_data_paths,
            dir_path,
            timer,
        )

        # print_seq(missing_input_data_items)

        if not missing_input_data_items:
            raise Exception("finished add?")
            return

        timer.push("init")
        # pq = faiss.read_index(empty_index_path)
        pq = faiss.IndexIDMap(faiss.read_index(empty_index_path))
        # pax(0, {"pq": pq})
        self.c_verbose(pq, True)
        timer.pop()

        # pax({"pq": pq})

        # timer.push("add")

        for item_index, missing_input_data_item in \
            enumerate(missing_input_data_items):

            batch_id, vec_id_start, input_data_path_item = missing_input_data_item

            # print_seq("%d, %d, %d, %s." % (
            #     item_index,
            #     batch_id,
            #     vec_id_start,
            #     input_data_path_item["residuals"],
            # ))

            timer.push("load-data")
            input_data_path = input_data_path_item["residuals"]
            input_data = utils.load_data([input_data_path], timer)["residuals"]
            nvecs = len(input_data)
            timer.pop()

            print_rank("pqs / add,  batch %d / %d. [ %d vecs ]" % (
                item_index,
                len(missing_input_data_items),
                nvecs,
            ))

            # print_seq(str(input_data.shape))

            timer.push("add")
            pq.add_with_ids(
                input_data,
                # list(range(vec_id_start, vec_id_start + nvecs)),
                np.arange(vec_id_start, vec_id_start + nvecs),
            )
            timer.pop()

            timer.push("write-partial")
            # faiss.write_index(pq, full_index_path)
            pq = self.write_partial_index(
                dir_path,
                pq,
                [ missing_input_data_item ],
                timer,
            )
            timer.pop()

        timer.push("merge-partials")
        self.merge_partial_indexes(dir_path, timer)
        timer.pop()

        # timer.pop()

# eof
