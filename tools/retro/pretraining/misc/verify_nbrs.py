# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import glob
import h5py
import os
import torch

from tools.retro.index import FaissBaseIndex, IndexFactory


def verify_neighbors(args, timer):

    timer.push("add-base")
    base_index = FaissBaseIndex(args)
    base_index_path = base_index.add(args.add_paths, args.index_dir_path, timer)
    timer.pop()

    timer.push("add-test")
    test_index = IndexFactory.get_index(args)
    test_index_path = test_index.add(args.add_paths, args.index_dir_path, timer)
    timer.pop()

    torch.distributed.barrier()

    if torch.distributed.get_rank() != 0:
        return

    timer.push("get-index-paths")
    # base_index = FaissBaseIndex(args)
    # test_index = IndexFactory.get_index(args)
    # indexes = [
    #     base_index,
    #     test_index,
    # ]
    # index_paths = [
    #     i.get_added_index_path(args.add_paths, args.index_dir_path)
    #     for i in indexes
    # ]
    index_paths = [
        base_index_path,
        test_index_path,
    ]
    index_names = [
        os.path.splitext(os.path.basename(p))[0]
        for p in index_paths
    ]
    timer.pop()

    neighbor_paths = [
        glob.glob(os.path.join(
            os.path.dirname(index_paths[0]),
            "neighbors",
            n,
            "*.hdf5",
        ))
        for n in index_names
    ]

    num_rows_checked = 0
    for base_neighbor_path in neighbor_paths[0]:

        def load_neighbors(path):
            f = h5py.File(path)
            neighbors = np.copy(f["neighbors"])
            f.close()
            return neighbors

        base_neighbor_name = os.path.basename(base_neighbor_path)
        test_neighbor_path = os.path.join(
            os.path.dirname(neighbor_paths[1][0]),
            base_neighbor_name,
        )

        if not os.path.isfile(test_neighbor_path):
            continue

        base_neighbors = load_neighbors(base_neighbor_path)
        test_neighbors = load_neighbors(test_neighbor_path)
        neighbors_equal = np.array_equal(base_neighbors, test_neighbors)

        assert neighbors_equal
        num_rows_checked += len(base_neighbors)

    assert num_rows_checked > 0, \
        "run 'query_index.sh/.py first; then run this script."
