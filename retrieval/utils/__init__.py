# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from collections import defaultdict
import h5py
import numpy as np
import os

from lutil import pax

from .timer import Timer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError as e:
        pass

def make_sub_dir(top_path, sub_name):
    sub_path = os.path.join(top_path, sub_name)
    mkdir(sub_path)
    return sub_path

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
        print("load path %d / %d ... '%s'." % (pi, len(paths), p), flush = True)
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
