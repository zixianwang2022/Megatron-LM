# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import glob
import h5py
import os
import torch

from tools.retro.index import FaissBaseIndex, IndexFactory


def verify_nbrs(args, timer):

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

    nbr_paths = [
        glob.glob(os.path.join(
            os.path.dirname(index_paths[0]),
            "nbrs",
            n,
            "*.hdf5",
        ))
        for n in index_names
    ]

    num_rows_checked = 0
    for base_nbr_path in nbr_paths[0]:

        def load_nbrs(path):
            f = h5py.File(path)
            nbrs = np.copy(f["neighbors"])
            f.close()
            return nbrs

        base_nbr_name = os.path.basename(base_nbr_path)
        test_nbr_path = os.path.join(
            os.path.dirname(nbr_paths[1][0]),
            base_nbr_name,
        )

        if not os.path.isfile(test_nbr_path):
            continue

        base_nbrs = load_nbrs(base_nbr_path)
        test_nbrs = load_nbrs(test_nbr_path)
        nbrs_equal = np.array_equal(base_nbrs, test_nbrs)

        assert nbrs_equal
        num_rows_checked += len(base_nbrs)

    assert num_rows_checked > 0, \
        "run 'query_index.sh/.py first; then run this script."
