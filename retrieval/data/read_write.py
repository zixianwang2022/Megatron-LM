# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from collections import defaultdict
import h5py
import numpy as np
import os

from lutil import pax, print_rank, print_seq

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def load_train_data(ntrain):
# def load_train_data(args):
# def load_data(args):
# def load_data(args, data_paths):
# def load_data(args):

#     pax({"args": args})

#     # ~~~~~~~~ load feats ~~~~~~~~
#     train_data = np.zeros((args.ntrain, args.nfeats), 'float32')
#     nloaded = 0
#     feat_paths = args.data_paths
#     for i, feat_path in enumerate(feat_paths):

#         f = h5py.File(feat_path, "r")
#         if 1:
#             d = np.copy(f["feat"])
#             i0 = nloaded
#             i1 = min(len(train_data), i0 + len(d))
#             d = d[:i1-i0]
#             if np.isnan(d).any():
#                 np.nan_to_num(d, copy = False, nan = 0.0)
#             try:
#                 train_data[i0:i1] = d
#             except:
#                 pax({
#                     "nloaded" : nloaded,
#                     "train_data" : str(train_data.shape),
#                     "d" : str(d.shape),
#                 })
#         else:
#             train_datas.append(f["feat"])
#         f.close()

#         nloaded += len(d)

#         print(
#             "load feat path %d / %d ... vecs %d." % (i, len(feat_paths), nloaded),
#             flush = True,
#         )

#         if nloaded >= args.ntrain:
#             break

#     args.ntrain = min(args.ntrain, nloaded)
#     train_data = train_data[:args.ntrain]

#     # pax({
#     #     # "train_datas" : [ a.shape for a in train_datas ],
#     #     "train_data / shape" : str(train_data.shape),
#     #     "train_data / dtype" : str(train_data.dtype),
#     #     "args" : args,
#     # })

#     return train_data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def save_data(data_map, dir_path, file_name):
def save_data(data_map, *args):

    # pax({"data_map": data_map})

    if len(args) == 1:
        path = args[0]
    elif len(args) == 2:
        dir_path, file_name = args
        path = os.path.join(dir_path, file_name)
    else:
        raise Exception("specialize for len(args) == %d." % len(args))

    if not os.path.isfile(path):
        f = h5py.File(path, "w")
        # f.create_dataset("data", data = input_data)
        for k, v in data_map.items():
            f.create_dataset(k, data = v)
        f.close()

    return path

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def load_data(paths):

#     # pax({"paths": paths})
#     count_map = defaultdict(lambda : 0)
#     for p in paths:
#         f = h5py.File(p, "r")
#         for k in f.keys():
#             count_map[k] += len(f[k])
#         f.close()

#     pax({
#         "paths" : paths,
#         "count_map" : count_map,
#     })

#     f = h5py.File(path, "r")
#     data_map = { k : np.copy(f[k]) for k in f.keys() }
#     f.close()

#     # pax({"data_map": data_map})

#     return data_map
def load_data(paths, timer):

    timer.push("shape")
    shape_map = defaultdict(lambda : (0, None))
    for p in paths:
        f = h5py.File(p, "r")
        for k in f.keys():
            shape = tuple(f[k].shape)
            shape_map[k] = (shape_map[k][0] + shape[0], shape[1])
        f.close()
    timer.pop()

    timer.push("alloc")
    data_map = { k : np.empty(s, dtype = "f4") for k, s in shape_map.items() }
    start_map = { k : 0 for k in shape_map }
    timer.pop()
    
    for pi, p in enumerate(paths):
        print_rank("load path %d / %d ... '%s'." % (pi, len(paths), p))
        timer.push("load")
        f = h5py.File(p, "r")
        for k in f.keys():
            i0 = start_map[k]
            i1 = i0 + len(f[k])
            data_map[k][i0:i1] = f[k]
            start_map[k] += len(f[k])
        f.close()
        timer.pop()

    # pax({
    #     "paths" : paths,
    #     "shape_map" : shape_map,
    #     "data_map" : { k : str(d.shape) for k, d in data_map.items() },
    #     "start_map" : start_map,
    # })

    return data_map

# eof
