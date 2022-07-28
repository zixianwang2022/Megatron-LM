# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import glob
import h5py
import numpy as np
import socket
import torch

from lutil import pax, print_rank, print_seq

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def get_data_paths(args, is_clean):
def get_all_data_paths(args, is_clean):

    hostname = socket.gethostname()
    # pax({"hostname": hostname})

    # ~~~~~~~~ feat paths [ hdf5 ] ~~~~~~~~
    if hostname.startswith("luna-"):
        if args.data_ty == "rand":
            raise Exception("update 'rand' for batch data loading; no 'ntrain'.")
            if 0:
                return np.random.rand(args.ntrain, 1024).astype("f4")
            else:
                from sklearn.datasets import make_blobs
                data, labels, centers = make_blobs(
                    n_samples = args.ntrain,
                    n_features = 1024,
                    centers = 32,
                    return_centers = True,
                )
                pax({
                    "data" : data,
                    "labels" : labels,
                    "centers" : centers,
                })
                return data
        elif args.data_ty == "corpus":
            # feat_paths = glob.glob("/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval/sampled_pretraining/*.feat.hdf5")
            feat_paths = glob.glob("/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval/corpus-%s.hdf5" % ("clean/*" if is_clean else "dirty/*.feat"))
        else:
            raise Exception("specialize for '%s'." % args.data_ty)

    elif hostname.startswith("rno"):
        # feat_paths = glob.glob(args.base_dir + "/enwiki-feat-16/*.hdf5")
        # feat_paths = glob.glob(args.base_dir + "/enwiki-feat-16-split/*.hdf5")
        # feat_paths = glob.glob(args.base_dir + "/enwiki-feat-1024/0000.hdf5")
        # feat_paths = glob.glob(args.base_dir + "/v2/data0/*feat.hdf5")
        if args.data_ty == "wiki":
            # feat_paths = glob.glob(args.base_dir + "/v2/data1/feat/*feat.hdf5") # matches banned doc_ids
            feat_paths = glob.glob(args.base_dir + "wiki/feat-%s/*feat.hdf5" % data_state) # matches banned doc_ids
        elif args.data_ty == "corpus":
            # feat_paths = glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/pretrain*feat.hdf5")
            if not is_clean:
                feat_paths = glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/*.feat.hdf5")
            else:
                feat_paths = glob.glob(args.base_dir+"/corpus-clean/*.hdf5")
            # feat_paths = glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/../boxinw/processed_data/chunks/sampled_pretraining/*.feat.hdf5")
        else:
            raise Exception("specialize for '%s'." % args.data_ty)

    elif hostname.startswith("ip-"):
        if args.data_ty == "wiki":
            feat_paths = glob.glob("/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/wiki/*.feat.hdf5")
        elif args.data_ty == "corpus":
            # feat_paths = glob.glob("/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/corpus/*.feat.hdf5")
            feat_paths = glob.glob("/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/corpus%s" % ("-dirty/*.feat.hdf5" if not is_clean else "-clean/*.hdf5"))
        else:
            raise Exception("specialize for '%s'." % args.data_ty)

    else:
        raise Exception("specialize for hostname '%s'." % hostname)

    feat_paths.sort()

    # args.data_paths = feat_paths

    # >>>
    if 0:
        n = 0
        for i, p in enumerate(feat_paths):
            if i % 20 == 0:
                print("counting feat path %d / %d." % (i, len(feat_paths)))
            f = h5py.File(p, "r")
            n += len(f["feat"])
        pax({
            "feat_paths" : feat_paths,
            "n" : n,
        })
    # <<<

    return feat_paths

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_train_add_data_paths(args, timer):

    all_paths = get_all_data_paths(args, True)

    # print_seq(all_paths)

    ntrain = None; train_paths = None
    nadd = None; add_paths = None
    ntotal = 0
    for path_index, path in enumerate(all_paths):
        f = h5py.File(path, "r")
        n = len(f["data"])
        f.close()

        ntotal += n

        if ntotal >= args.ntrain and ntrain is None:
            ntrain = ntotal
            train_paths = list(all_paths[:(path_index+1)])
        if ntotal >= args.nadd and nadd is None:
            nadd = ntotal
            add_paths = list(all_paths[:(path_index+1)])

        if ntrain is not None and nadd is not None:
            break

    if ntrain is None or nadd is None:
        raise Exception("not even data paths?")

    # pax(0, {
    #     "all_paths" : all_paths,
    #     "ntrain" : ntrain,
    #     "nadd" : nadd,
    #     "train_paths" : train_paths,
    #     "add_paths" : add_paths,
    # })

    return ntrain, nadd, train_paths, add_paths

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
import os

def _clean_data(args, timer):

    # raise Exception("clean again?")

    assert torch.distributed.get_rank() == 0

    # >>>
    if 0:
        # filename = "0038__0010-01500000.hdf5"
        filename = "0039__0010-02500000.hdf5"
        f = h5py.File(os.path.join(args.base_dir, "corpus-clean", filename))
        # pax({
        #     "data": f["data"],
        #     "keys" : list(f.keys()),
        # })
        d = np.copy(f["data"])
        f.close()
        pax({"filename": filename, "d": str(d.shape)})
    # <<<

    batch_size = int(1e6)
    batch = np.zeros((batch_size, args.nfeats), "f4")
    b0 = 0

    # num_batches = 0
    def save_batch(dirty_index, d1): # batch, b0, num_batches):
        nonlocal b0
        nonlocal num_batches

        if b0 == 0:
            return

        filename = "%04d__%04d-%08d.hdf5" % (num_batches, dirty_index, d1)
        clean_path = os.path.join(args.base_dir, "corpus-clean", filename)
        print("saving '%s'. [ %d samples ]" % (filename, b0))
        # pax({"clean_path": clean_path})
        f = h5py.File(clean_path, "w")
        f.create_dataset("data", data = batch[:b0])
        f.close()

        b0 = 0
        num_batches += 1

        # >>>
        # print("bye."); exit(0) # tmp, for batch 0039, 2.5M-3.5M
        # <<<

    def get_dirty_start_index(clean_path):
        # >>>
        f = h5py.File(clean_path, "r")
        shape = f["data"].shape
        f.close()
        assert shape[0] > 0 and shape[1] == 1024
        # pax({"shape": shape})
        # <<<
        return [
            int(a)
            for a in clean_path.split("__")[1].split(".")[0].split("-")
        ]

    dirty_paths = get_all_data_paths(args, False)
    clean_paths = get_all_data_paths(args, True)

    # pax(0, {
    #     "dirty_paths" : dirty_paths,
    #     "clean_paths" : clean_paths,
    # })

    if 1:
        if not clean_paths:
            dirty_start_index, d0 = 0, 0
        else:
            dirty_start_index, d0 = get_dirty_start_index(clean_paths[-1])
        num_batches = len(clean_paths)
    else:
        # raise Exception("stop.")
        num_batches = 39
        dirty_start_index, d0 = get_dirty_start_index(os.path.join(
            args.base_dir,
            "corpus-clean",
            "0038__0010-01500000.hdf5",
        ))

    # pax({
    #     "args" : args,
    #     "dirty_paths" : dirty_paths,
    #     "clean_paths" : clean_paths,
    #     "num_batches" : num_batches,
    #     "dirty_start_index" : dirty_start_index,
    #     "d0" : d0,
    # })

    # for i, dirty_path in enumerate(dirty_paths):
    for dirty_index in range(dirty_start_index, len(dirty_paths)):

        dirty_path = dirty_paths[dirty_index]

        print("load feat path %d / %d." % (
            dirty_index,
            len(dirty_paths),
        ), flush = True)

        f = h5py.File(dirty_path, "r")
        d = np.copy(f["feat"])
        # d = f["feat"]
        if np.isnan(d).any():
            np.nan_to_num(d, copy = False, nan = 0.0)
        f.close()

        # d0 = 0
        while d0 < len(d):
            d1 = min(len(d), d0 + batch_size - b0)
            batch[b0:(b0+d1-d0)] = d[d0:d1]
            b0 += d1 - d0
            if b0 == batch_size:
                save_batch(dirty_index, d1)
            elif b0 > batch_size:
                raise Exception("something's wrong.")
            # else:
            #     pax({
            #         "b0" : b0,
            #         "d0" : d0,
            #         "d1" : d1,
            #     })
            d0 = d1
        d0 = 0

    save_batch(len(dirty_paths) - 1, d1)

def clean_data(args, timer):

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        _clean_data(args, timer)

    torch.distributed.barrier()

    exit(0)

# eof
