# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import glob
import h5py
# import numpy as np
import socket
# import torch

from lutil import pax, print_rank, print_seq

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def get_data_paths(args, is_clean):
def get_all_data_paths(args, is_clean = True):

    hostname = socket.gethostname()
    # pax({"hostname": hostname})

    # ~~~~~~~~ feat paths [ hdf5 ] ~~~~~~~~
    if hostname.startswith("luna-"):
        # if args.data_ty == "rand":
        #     raise Exception("update 'rand' for batch data loading; no 'ntrain'.")
        #     if 0:
        #         return np.random.rand(args.ntrain, 1024).astype("f4")
        #     else:
        #         from sklearn.datasets import make_blobs
        #         data, labels, centers = make_blobs(
        #             n_samples = args.ntrain,
        #             n_features = 1024,
        #             centers = 32,
        #             return_centers = True,
        #         )
        #         pax({
        #             "data" : data,
        #             "labels" : labels,
        #             "centers" : centers,
        #         })
        #         return data
        if args.data_ty == "rand":
            feat_paths = glob.glob("/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval/data/%s/*.hdf5" % args.data_ty)
            pax(0, {"feat_paths": feat_paths})
        elif args.data_ty == "corpus":
            # feat_paths = glob.glob("/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval/sampled_pretraining/*.feat.hdf5")
            feat_paths = glob.glob("/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval/data/corpus-%s.hdf5" % ("clean/*" if is_clean else "dirty/*.feat"))
        else:
            raise Exception("specialize for '%s'." % args.data_ty)

    elif hostname.startswith("rno") or "dracocpu" in hostname:
        # feat_paths = glob.glob(args.base_dir + "/enwiki-feat-16/*.hdf5")
        # feat_paths = glob.glob(args.base_dir + "/enwiki-feat-16-split/*.hdf5")
        # feat_paths = glob.glob(args.base_dir + "/enwiki-feat-1024/0000.hdf5")
        # feat_paths = glob.glob(args.base_dir + "/v2/data0/*feat.hdf5")
        if args.data_ty == "wiki":
            # feat_paths = glob.glob(args.base_dir + "/v2/data1/feat/*feat.hdf5") # matches banned doc_ids
            feat_paths = glob.glob(args.base_dir + "/data/wiki/feat-%s/*feat.hdf5" % data_state) # matches banned doc_ids
        elif args.data_ty == "corpus":
            # feat_paths = glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/pretrain*feat.hdf5")
            if not is_clean:
                feat_paths = glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/sampled_pretraining/*.feat.hdf5")
            else:
                feat_paths = glob.glob(args.base_dir+"/data/corpus-clean/*.hdf5")
            # feat_paths = glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/../boxinw/processed_data/chunks/sampled_pretraining/*.feat.hdf5")
        elif args.data_ty.startswith("rand-"):
            feat_paths = glob.glob(args.base_dir + "/data/%s/*.hdf5" % args.data_ty)
        else:
            raise Exception("specialize for '%s'." % args.data_ty)

    elif hostname.startswith("ip-"):
        if args.data_ty == "wiki":
            feat_paths = glob.glob("/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/data/wiki/*.feat.hdf5")
        elif args.data_ty == "corpus":
            # feat_paths = glob.glob("/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/corpus/*.feat.hdf5")
            feat_paths = glob.glob("/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/data/corpus%s" % ("-dirty/*.feat.hdf5" if not is_clean else "-clean/*.hdf5"))
        # elif args.data_ty == "rand-100k":
        #     feat_paths = glob.glob("/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/data/rand-100k/*.hdf5")
        elif args.data_ty.startswith("rand-"):
            feat_paths = glob.glob("/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/data/%s/*.hdf5" % args.data_ty)
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
                print_rank(0, "counting feat path %d / %d." % (i, len(feat_paths)))
            f = h5py.File(p, "r")
            n += len(f["data"]) # feat"])
        pax(0, {
            "feat_paths" : feat_paths,
            "n" : n,
        })
    # <<<

    return feat_paths

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def get_train_add_data_paths(args, timer):
def get_train_add_data_paths(args):

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

# eof
