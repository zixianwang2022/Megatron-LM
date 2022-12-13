# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from collections import defaultdict
import h5py
import json
import numpy as np
import os
import torch


def count_nvecs(base_path, index_path):

    # Neighbor paths.
    nbr_paths = [
        p
        for p in os.listdir(os.path.join(base_path, index_path))
        if p.endswith(".hdf5")
    ]
    nbr_paths.sort() # ... necessary?

    # Vector count.
    nvecs = 0
    for nbr_path_index, nbr_path in enumerate(nbr_paths):
        f = h5py.File(os.path.join(base_path, index_path, nbr_path), "r")
        nvecs += len(f["neighbors"])
        f.close()

    return nvecs


def find_missing_nbr_paths(base_path, index_path_0, index_path_1):

    # Neighbor paths.
    nbr_paths = [
        p
        for p in os.listdir(os.path.join(base_path, index_path_0))
        if p.endswith(".hdf5")
    ]
    nbr_paths.sort() # ... necessary?

    # Missing paths.
    missing_nbr_paths = []
    for nbr_path_index, nbr_path in enumerate(nbr_paths):
        if not os.path.exists(os.path.join(base_path, index_path_1, nbr_path)):
            missing_nbr_paths.append(nbr_path)


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


def get_acc_map(base_path, nnbrs, index_path):

    flat_nbr_path = "Flat__t65191936__neighbors.hdf5"

    # ~~~~~~~~ nbr paths ~~~~~~~~
    index_nbr_paths = [
        p
        for p in os.listdir(os.path.join(base_path, index_path))
        if p.endswith(".hdf5")
    ]
    index_nbr_paths.sort() # ... unnecessary

    # ~~~~~~~~ load flat nbrs ~~~~~~~~
    f = h5py.File(os.path.join(base_path, flat_nbr_path), "r")
    flat_nbr_grid = np.copy(f["neighbors"])
    f.close()

    # ~~~~~~~~ load index nbrs ~~~~~~~~
    index_nbr_grids = []
    nloaded = 0
    for index_nbr_path in index_nbr_paths:

        f = h5py.File(os.path.join(base_path, index_path, index_nbr_path), "r")
        index_nbr_grid = np.copy(f["neighbors"])
        index_nbr_grids.append(index_nbr_grid)
        nloaded += len(index_nbr_grid)
        f.close()

        if nloaded >= len(flat_nbr_grid):
            break

    index_nbr_grid = np.concatenate(index_nbr_grids, axis = 0)
    index_nbr_grid = index_nbr_grid[:len(flat_nbr_grid)]

    # ~~~~~~~~ acc map ~~~~~~~~
    acc_map = {}
    for nnbr_index, nnbr in enumerate(nnbrs):
        print("  nnbr %d [ %d / %d ]." % (nnbr, nnbr_index, len(nnbrs)))
        overlaps = rowwise_intersection(
            flat_nbr_grid[:, :nnbr],
            index_nbr_grid[:, :nnbr],
        )
        acc_map[nnbr] = np.mean(overlaps) / nnbr

    return acc_map


def vis_acc(index_paths, nnbrs):

    assert torch.distributed.get_rank() == 0

    # ~~~~~~~~ acc map ~~~~~~~~
    acc_map = {}
    for k, index_path in enumerate(index_paths):
        print("index %d / %d ... '%s'." % (k, len(index_paths), index_path))
        # acc_map[index_path] = get_acc_map(base_path, nnbrs, index_path)
        acc_map[index_path] = get_acc_map(index_path, nnbrs)

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
    nnbrs = [ 1, 2, 5, 10 ]
    vis_acc(index_paths, nnbrs)
    timer.pop()
