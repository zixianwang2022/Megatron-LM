# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import argparse
from collections import defaultdict
import h5py
import json
import numpy as np
import os

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def load_nbr_map(base_path, index_path):
def count_nvecs(base_path, index_path):

    # ~~~~~~~~ nbr paths ~~~~~~~~
    nbr_paths = [
        p
        for p in os.listdir(os.path.join(base_path, index_path))
        if p.endswith(".hdf5")
    ]
    nbr_paths.sort() # ... unnecessary

    # ~~~~~~~~ nvecs ~~~~~~~~
    nvecs = 0
    for nbr_path_index, nbr_path in enumerate(nbr_paths):
        f = h5py.File(os.path.join(base_path, index_path, nbr_path), "r")
        # nbrs = np.array(f["neighbors"])
        # pax({
        #     "f" : f,
        #     "f / keys" : list(f.keys()),
        #     "nbrs" : nbrs,
        # })
        nvecs += len(f["neighbors"])
        # print("nbr path %d / %d ... nvecs %d." % (
        #     nbr_path_index,
        #     len(nbr_paths),
        #     nvecs,
        # ))
        f.close()

    # ~~~~~~~~ debug ~~~~~~~~
    # pax({
    #     "nbr_paths" : nbr_paths,
    #     "nvecs" : nvecs,
    # })

    # ~~~~~~~~ return ~~~~~~~~
    return nvecs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_missing_nbr_paths(base_path, index_path_0, index_path_1):

    # ~~~~~~~~ nbr paths ~~~~~~~~
    nbr_paths = [
        p
        for p in os.listdir(os.path.join(base_path, index_path_0))
        if p.endswith(".hdf5")
    ]
    nbr_paths.sort() # ... unnecessary

    # ~~~~~~~~ nvecs ~~~~~~~~
    missing_nbr_paths = []
    for nbr_path_index, nbr_path in enumerate(nbr_paths):
        if not os.path.exists(os.path.join(base_path, index_path_1, nbr_path)):
            missing_nbr_paths.append(nbr_path)

    # ~~~~~~~~ debug ~~~~~~~~
    pax({"missing_nbr_paths": missing_nbr_paths})

    # ~~~~~~~~ return ~~~~~~~~
    # ?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def intersect1d_padded(x):
#     x, y = np.split(x, 2)
#     # pax({"x": x.shape, "y": y.shape})
#     # padded_intersection = -1 * np.ones(x.shape, dtype=np.int)
#     # intersection = np.intersect1d(x, y)
#     # padded_intersection[:intersection.shape[0]] = intersection
#     # return padded_intersection
#     return len(np.intersect1d(x, y))

def rowwise_intersection(a, b):
    # pax({"a": a.shape, "b": b.shape})
    return np.apply_along_axis(
        # intersect1d_padded,
        lambda a : len(np.intersect1d(*np.split(a, 2))),
        1,
        np.concatenate((a, b), axis = 1),
    )

# def compare_nbrs(base_path, index_path_0, index_path_1, nnbrs):

#     # ~~~~~~~~ nbr paths ~~~~~~~~
#     nbr_paths = [
#         p
#         for p in os.listdir(os.path.join(base_path, index_path_0))
#         if p.endswith(".hdf5")
#     ]
#     nbr_paths.sort() # ... unnecessary

#     # ~~~~~~~~ acc map ~~~~~~~~
#     # with open() as f:
#     acc_path = os.path.join(base_path, index_path_1, "accuracy.jsonl")
#     existing_acc_keys = set()
#     if os.path.isfile(acc_path):
#         with open(acc_path) as f:
#             for line in f.read().splitlines():
#                 acc_map = json.loads(line)
#                 existing_acc_keys.add(acc_map["nbr_path"])
#     # if os.path.isfile(acc_path):
#     #     raise Exception("hi.")
#     #     existing_acc_keys = set(json.load(acc_path).keys())
#     # else:
#     #     existing_acc_keys = set()

#     # ~~~~~~~~ nvecs ~~~~~~~~
#     for nbr_path_index, nbr_path in enumerate(nbr_paths):

#         if nbr_path in existing_acc_keys:
#             continue
#         existing_acc_keys.add(nbr_path)

#         f0 = h5py.File(os.path.join(base_path, index_path_0, nbr_path), "r")
#         f1 = h5py.File(os.path.join(base_path, index_path_1, nbr_path), "r")
#         nbr_grid_0 = f0["neighbors"]
#         nbr_grid_1 = f1["neighbors"]
#         assert nbr_grid_0.shape == nbr_grid_1.shape
#         nvecs = len(nbr_grid_0)
#         # >>>
#         # for r in range(len(nbr_grid_0)):
#         #     if r % 10000 == 0:
#         #         print("nbr path %d / %d, row %d / %d." % (
#         #             nbr_path_index,
#         #             len(nbr_paths),
#         #             r,
#         #             len(nbr_grid_0),
#         #         ))
#         #     # >>>
#         #     # for nnbr in nnbrs:
#         #     #     nbrs_0 = set(nbr_grid_0[r][:nnbr])
#         #     #     nbrs_1 = set(nbr_grid_1[r][:nnbr])
#         #     #     overlap_map[nnbr].append(len(nbrs_0 & nbrs_1))
#         #     # +++
#         #     for nnbr in nnbrs:
#         #         nbrs_0 = nbr_grid_0[r][:nnbr]
#         #         nbrs_1 = nbr_grid_1[r][:nnbr]
#         #         overlap_map[nnbr].append(len(np.intersect1d(nbrs_0, nbrs_1)))
#         #         # pax({"overlap_map": overlap_map})
#         #     # <<<
#         # +++
#         overlap_map = defaultdict(list)
#         for nnbr in nnbrs:
#             print("nbr path %d / %d ... nnbr %d." % (
#                 nbr_path_index,
#                 len(nbr_paths),
#                 nnbr,
#             ), flush = True)
#             overlaps = rowwise_intersection(
#                 nbr_grid_0[:, :nnbr],
#                 nbr_grid_1[:, :nnbr],
#             )
#             overlap_map[nnbr].extend(overlaps)
#             # pax({"result": result})
#         # <<<
#         f0.close()
#         f1.close()

#         acc_map = {k : np.mean(v) / k for k, v in overlap_map.items()}
#         with open(acc_path, "a") as f:
#             f.write(json.dumps({
#                 "nbr_path" : nbr_path,
#                 "count" : nvecs,
#                 "acc" : acc_map,
#             }) + "\n")

#         # >>>
#         # pax({"overlap_map": overlap_map})
#         # if nbr_path_index == 1:
#         #     break
#         # <<<

#     # ~~~~~~~~ debug ~~~~~~~~
#     pax({
#         # "nbr_paths" : nbr_paths,
#         "overlap_map" : overlap_map,
#         "acc_map" : {k : np.mean(v) / k for k, v in overlap_map.items()},
#     })

#     # ~~~~~~~~ return ~~~~~~~~
#     # ?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def load_acc_map(base_path, index_path):

#     # ~~~~~~~~ parse jsonl ~~~~~~~~
#     acc_path = os.path.join(base_path, index_path, "accuracy.jsonl")
#     acc_map = defaultdict(list)
#     count_map = defaultdict(lambda : 0)
#     with open(acc_path) as f:
#         for line in f.read().splitlines():
#             entry = json.loads(line)
#             # pax({"entry": entry})
#             for n, acc in entry["acc"].items():
#                 acc_map[int(n)].append(acc * entry["count"])
#                 count_map[int(n)] += entry["count"]

#     # ~~~~~~~~ normalize accs ~~~~~~~~
#     acc_map = { k : sum(v) / count_map[k] for k, v in acc_map.items() }

#     # ~~~~~~~~ debug ~~~~~~~~
#     # pax({
#     #     # "count_map" : count_map,
#     #     "acc_map" : acc_map,
#     # })

#     # ~~~~~~~~ return ~~~~~~~~
#     return acc_map

# # def vis_acc(base_path, index_path):
# def vis_acc(base_path):

#     index_paths = [
#         "OPQ64_64,IVF4194304_HNSW32,PQ64__t66630804",
#         "OPQ32_128,IVF4194304_HNSW32,PQ32__t66630804",
#         "OPQ32_64,IVF4194304_HNSW32,PQ32__t66630804",
#         "OPQ16_128,IVF4194304_HNSW32,PQ16__t66630804",
#     ]

#     # ~~~~~~~~ acc map ~~~~~~~~
#     acc_map = { i : load_acc_map(base_path, i) for i in index_paths }

#     # ~~~~~~~~ vert map ~~~~~~~~
#     vert_map = {}
#     for i, m in acc_map.items():
#         verts = list(m.items())
#         verts.sort(key = lambda v : v[0])
#         vert_map[i] = verts

#     # ~~~~~~~~ plot ~~~~~~~~
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     for i, vs in vert_map.items():
#         # pax({"vs": vs})
#         # x, y = zip(*vs)
#         # pax({"x": x, "y": y})
#         plt.plot(*zip(*vs), label = i)
#     plt.legend()
#     plt.savefig("accs.png")

#     # ~~~~~~~~ debug ~~~~~~~~
#     pax({
#         "index_paths" : index_paths,
#         "acc_map" : acc_map,
#         "vert_map" : vert_map,
#     })

#     # ~~~~~~~~ return ~~~~~~~~
#     # ?

# def compare_nbrs(base_path, index_path, nnbrs):
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

    # ~~~~~~~~ debug ~~~~~~~~
    # pax({
    #     # "flat_nbr_grid" : flat_nbr_grid,
    #     # "index_nbr_grid" : index_nbr_grid,
    #     "overlaps" : overlaps,
    #     "acc_map" : acc_map,
    # })

    # ~~~~~~~~ return ~~~~~~~~
    return acc_map

def vis_acc(base_path, nnbrs):

    index_paths = [
        "OPQ64_128,IVF4194304_HNSW32,PQ64__t65191936",
        "OPQ64_256,IVF4194304_HNSW32,PQ64__t65191936",
        "OPQ64_512,IVF4194304_HNSW32,PQ64__t65191936",

        "OPQ32_128,IVF4194304_HNSW32,PQ32__t65191936",
        "OPQ32_256,IVF4194304_HNSW32,PQ32__t65191936",
        "OPQ32_512,IVF4194304_HNSW32,PQ32__t65191936",

        "OPQ16_128,IVF4194304_HNSW32,PQ16__t65191936",
        "OPQ16_256,IVF4194304_HNSW32,PQ16__t65191936",
        "OPQ16_512,IVF4194304_HNSW32,PQ16__t65191936",
    ]

    # ~~~~~~~~ acc map ~~~~~~~~
    acc_map = {}
    for k, index_path in enumerate(index_paths):
        print("index %d / %d ... '%s'." % (k, len(index_paths), index_path))
        acc_map[index_path] = get_acc_map(base_path, nnbrs, index_path)

    # pax({"acc_map": acc_map})

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
        # pax({"vs": vs})
        # x, y = zip(*vs)
        # pax({"x": x, "y": y})
        plt.plot(*zip(*vs), label = i.split(",")[0])
    plt.legend()
    plt.savefig("accs.png")

    # ~~~~~~~~ debug ~~~~~~~~
    pax({
        "index_paths" : index_paths,
        "acc_map" : acc_map,
        "vert_map" : vert_map,
    })

    # ~~~~~~~~ return ~~~~~~~~
    # ?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required = True, choices = [
        # "sandbox",
        "test-knn",
        "proc",
        "vis",
    ])
    # parser.add_argument("--i0", required = True)
    # parser.add_argument("--i1", required = True)
    # parser.add_argument("--index", required = True)
    args = parser.parse_args()
    args.base_path = \
        "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/v2/n2000"
    # args.nnbrs = 1, 2, 5, 10, 20, 50, 100, 200
    # args.nnbrs = 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000
    args.nnbrs = 1, 2, 5, 10
    # args.nnbrs = 2, 20, 200
    # args.nnbrs = 200,

    # pax({"args": args})

    # index_nvec_map = {}
    # for index_path in index_paths:
    #     nvecs = count_nvecs(base_path, index_path)
    #     index_nvec_map[index_path] = nvecs
    #     print("'%s' ... nvecs %d." % (index_path, nvecs))

    # find_missing_nbr_paths(
    #     base_path,
    #     "OPQ64_128,IVF4194304_HNSW32,PQ64__t66630804/",
    #     "OPQ64_64,IVF4194304_HNSW32,PQ64__t66630804/",
    # )
    # raise Exception("hi.")

    if args.task == "proc":
        # compare_nbrs(
        #     args.base_path,
        #     args.i0,
        #     args.i1,
        #     args.nnbrs,
        # )
        # compare_nbrs(
        #     args.base_path,
        #     args.i1,
        #     args.nnbrs,
        # )
        # compare_nbrs(args)
        raise Exception("in flux.")
    elif args.task == "vis":
        # vis_acc(args.base_path) # , args.i1)
        vis_acc(args.base_path, args.nnbrs)
    # elif args.task == "test-knn":
    #     test_knn()
    elif args.task == "query-flat":
        query_flat_nns()
    else:
        raise Exception("specialize for task '%s'." % args.task)

    pax({
        "base_path" : base_path,
        "index_paths" : index_paths,
        # "index_nvec_map" : index_nvec_map,
    })

# eof
