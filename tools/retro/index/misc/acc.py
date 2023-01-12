# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import argparse
from collections import defaultdict
import h5py
import json
import numpy as np
import os
import torch


def count_nvecs(base_path, index_path):

    # Neighbor paths.
    neighbor_paths = [
        p
        for p in os.listdir(os.path.join(base_path, index_path))
        if p.endswith(".hdf5")
    ]
    neighbor_paths.sort() # ... necessary?

    # Vector count.
    nvecs = 0
    for neighbor_path_index, neighbor_path in enumerate(neighbor_paths):
        f = h5py.File(os.path.join(base_path, index_path, neighbor_path), "r")
        nvecs += len(f["neighbors"])
        f.close()

    return nvecs


def find_missing_neighbor_paths(base_path, index_path_0, index_path_1):

    # Neighbor paths.
    neighbor_paths = [
        p
        for p in os.listdir(os.path.join(base_path, index_path_0))
        if p.endswith(".hdf5")
    ]
    neighbor_paths.sort() # ... necessary?

    # Missing paths.
    missing_neighbor_paths = []
    for neighbor_path_index, neighbor_path in enumerate(neighbor_paths):
        if not os.path.exists(
                os.path.join(base_path, index_path_1, neighbor_path)):
            missing_neighbor_paths.append(neighbor_path)


# def intersect1d_padded(x):
#     x, y = np.split(x, 2)
#     # padded_intersection = -1 * np.ones(x.shape, dtype=np.int)
#     # intersection = np.intersect1d(x, y)
#     # padded_intersection[:intersection.shape[0]] = intersection
#     # return padded_intersection
#     return len(np.intersect1d(x, y))


def rowwise_intersection(a, b):
    return np.apply_along_axis(
        # intersect1d_padded,
        lambda a : len(np.intersect1d(*np.split(a, 2))),
        1,
        np.concatenate((a, b), axis = 1),
    )


def get_acc_map(base_path, num_neighbors_list, index_path):

    flat_neighbor_path = "Flat__t65191936__neighbors.hdf5"

    # ~~~~~~~~ neighbor paths ~~~~~~~~
    index_neighbor_paths = [
        p
        for p in os.listdir(os.path.join(base_path, index_path))
        if p.endswith(".hdf5")
    ]
    index_neighbor_paths.sort() # ... unnecessary

    # ~~~~~~~~ load flat neighbors ~~~~~~~~
    f = h5py.File(os.path.join(base_path, flat_neighbor_path), "r")
    flat_neighbor_grid = np.copy(f["neighbors"])
    f.close()

    # ~~~~~~~~ load index neighbors ~~~~~~~~
    index_neighbor_grids = []
    nloaded = 0
    for index_neighbor_path in index_neighbor_paths:

        f = h5py.File(os.path.join(
            base_path, index_path, index_neighbor_path), "r")
        index_neighbor_grid = np.copy(f["neighbors"])
        index_neighbor_grids.append(index_neighbor_grid)
        nloaded += len(index_neighbor_grid)
        f.close()

        if nloaded >= len(flat_neighbor_grid):
            break

    index_neighbor_grid = np.concatenate(index_neighbor_grids, axis = 0)
    index_neighbor_grid = index_neighbor_grid[:len(flat_neighbor_grid)]

    # ~~~~~~~~ acc map ~~~~~~~~
    acc_map = {}
    for num_neighbor_index, num_neighbors in enumerate(num_neighbors_list):
        print("  num neighbors %d [ %d / %d ]." % (
            num_neighbors,
            num_neighbor_index,
            len(num_neighbors_list),
        ))
        overlaps = rowwise_intersection(
            flat_neighbor_grid[:, :num_neighbors],
            index_neighbor_grid[:, :num_neighbors],
        )
        acc_map[num_neighbors] = np.mean(overlaps) / num_neighbors

    return acc_map


def vis_acc(index_paths, num_neighbors_list):

    assert torch.distributed.get_rank() == 0

    # ~~~~~~~~ acc map ~~~~~~~~
    acc_map = {}
    for k, index_path in enumerate(index_paths):
        print("index %d / %d ... '%s'." % (k, len(index_paths), index_path))
        # acc_map[index_path] = get_acc_map(base_path, num_neighbors_list, index_path)
        acc_map[index_path] = get_acc_map(index_path, num_neighbors_list)

    # ~~~~~~~~ vert map ~~~~~~~~
    vert_map = {}
    for i, m in acc_map.items():
        verts = list(m.items())
        verts.sort(key = lambda v : v[0])
        vert_map[i] = verts

    # ~~~~~~~~ plot ~~~~~~~~
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i, vs in vert_map.items():
        # x, y = zip(*vs)
        plt.plot(*zip(*vs), label = i.split(",")[0])
    plt.legend()
    plt.savefig("accs.png")


def plot_query_acc():

    if torch.distributed.get_rank() != 0:
        return

    timer = Timer()

    timer.push("get-index-paths")
    base_index = FaissBaseIndex(args)
    test_index = IndexFactory.get_index(args)
    base_index_path = base_index.get_added_index_path(
        args.train_paths,
        args.index_dir_path,
    )
    test_index_path = test_index.get_added_index_path(
        args.train_paths,
        args.index_dir_path,
    )
    index_paths = [
        base_index_path,
        test_index_path,
    ]
    timer.pop()

    timer.push("vis-acc")
    num_neighbors_list = [ 1, 2, 5, 10 ]
    vis_acc(index_paths, num_neighbors_list)
    timer.pop()
